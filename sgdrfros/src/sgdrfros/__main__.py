import rospy
import argparse
import sys
from typing import Any, Optional, Union, Tuple, List

import pyro
import pyro.util
import torch
from geometry_msgs.msg import Point

from sgdrf_msgs.msg import CategoricalObservation
from sgdrf_srvs.srv import (
    TopicProb,
    TopicProbResponse,
    WordProb,
    WordProbResponse,
    WordTopicMatrix,
    WordTopicMatrixResponse,
)
from std_msgs.msg import Float64

from .kernel import KernelType
from .model import SGDRF
from .optimizer import OptimizerType
from .subsample import SubsampleType

RANGETYPE = Tuple[Union[int, float], Union[int, float], Union[int, float]]


class SGDRFNode:
    def __init__(self):
        rospy.init_node()
        self.parser = argparse.ArgumentParser()
        self.setup_parameters()
        self.sgdrf = self.initialize_sgdrf()
        self.obs_subscriber = rospy.Subscriber(
            f"categorical_observation__{self.sgdrf.V}__",
            CategoricalObservation,
            self.new_obs_callback,
        )
        self.loss_publisher = rospy.Publisher("loss", Float64, queue_size=10)
        self.topic_prob_server = rospy.Service(
            "topic_prob", TopicProb, "topic_prob", self.topic_prob_service_callback
        )
        self.word_prob_server = rospy.Service(
            "word_prob", WordProb, self.word_prob_service_callback
        )
        self.word_topic_matrix_server = rospy.Service(
            "word_topic_matrix",
            WordTopicMatrix,
            self.word_topic_matrix_service_callback,
        )

    def spin(self, rate: int = 10):
        sleep_rate = rospy.Rate(rate)
        while not rospy.is_shutdown():
            self.training_step_callback()
            rate.sleep()

    def param_name(self, name: str) -> str:
        return ".".join([rospy.get_name(), name])

    def param(self, name: str) -> str:
        return rospy.get_param(self.param_name(name))

    def init_random_seed(self):
        random_seed = self.param("random_seed")
        pyro.util.set_rng_seed(random_seed)

    def initialize_sgdrf(self):
        dims = self.param("dims")
        xu_ns = self.param("xu_ns")
        d_mins = self.param("d_mins")
        d_maxs = self.param("d_maxs")
        if len(d_maxs) == 1:
            d_maxs = d_maxs * dims

        assert all(
            x < y for x, y in zip(d_mins, d_maxs)
        ), "d_mins must be smaller than d_maxs in every dimension"

        V = self.param("V")
        K = self.param("K")
        max_obs = self.param("max_obs")

        dir_p = self.param("dir_p")
        assert dir_p > 0, "dir_p must be greater than zero"
        kernel_type_string = self.param("kernel_type")
        assert (
            kernel_type_string in KernelType._member_names_
        ), "kernel_type must be one of " + str(KernelType._member_names_)
        kernel_type = KernelType[kernel_type_string]
        kernel_lengthscale = self.param("kernel_lengthscale")
        assert kernel_lengthscale > 0, "kernel_lengthscale must be greater than zero"
        kernel_variance = self.param("kernel_variance")
        assert kernel_variance > 0, "kernel_variance must be greater than zero"
        optimizer_type_string = self.param("optimizer_type")
        assert (
            optimizer_type_string in OptimizerType._member_names_
        ), "optimizer_type must be one of " + str(OptimizerType._member_names_)
        optimizer_type = OptimizerType[optimizer_type_string]
        optimizer_lr = self.param("optimizer_lr")
        assert optimizer_lr > 0, "optimizer_lr must be greater than zero"
        optimizer_clip_norm = self.param("optimizer_clip_norm")
        assert optimizer_clip_norm > 0, "optimizer_lr must be greater than zero"
        device_string = self.param("device")
        device = torch.device(device_string)
        subsample_n = self.param("subsample_n")
        subsample_type_string = self.param("subsample_type")
        assert (
            subsample_type_string in SubsampleType._member_names_
        ), "subsample_type must be one of " + str(SubsampleType._member_names_)
        subsample_type = SubsampleType[subsample_type_string]
        subsample_weight = self.param("subsample_weight")
        subsample_exp = self.param("subsample_exp")
        assert subsample_exp > 0, "subsample_exp must be greater than zero"
        whiten = self.param("whiten")
        fail_on_nan_loss = self.param("fail_on_nan_loss")
        num_particles = self.param("num_particles")
        jit = self.param("jit")
        subsample_params = {
            "subsample_weight": subsample_weight,
            "subsample_exp": subsample_exp,
        }
        sgdrf = SGDRF(
            xu_ns=xu_ns,
            d_mins=d_mins,
            d_maxs=d_maxs,
            V=V,
            K=K,
            max_obs=max_obs,
            dir_p=dir_p,
            kernel_type=kernel_type,
            kernel_lengthscale=kernel_lengthscale,
            kernel_variance=kernel_variance,
            optimizer_type=optimizer_type,
            optimizer_lr=optimizer_lr,
            optimizer_clip_norm=optimizer_clip_norm,
            device=device,
            subsample_n=subsample_n,
            subsample_type=subsample_type,
            subsample_params=subsample_params,
            whiten=whiten,
            fail_on_nan_loss=fail_on_nan_loss,
            num_particles=num_particles,
            jit=jit,
        )
        return sgdrf

    def categorical_observation_to_tensors(self, msg: CategoricalObservation):
        pose_stamped = msg.pose_stamped
        obs = msg.obs
        assert (
            len(obs) == self.sgdrf.V
        ), "message observation length does not match SGDRF initialized vocabulary size"
        pose = pose_stamped.pose
        point = pose.position
        x_raw = [point.x]
        if self.sgdrf.dims >= 2:
            x_raw.append([point.y])
        if self.sgdrf.dims == 3:
            x_raw.append([point.z])
        xs = torch.tensor(x_raw, dtype=torch.float, device=self.sgdrf.device).unsqueeze(
            0
        )
        ws = torch.tensor(obs, dtype=torch.int, device=self.sgdrf.device).unsqueeze(0)
        return xs, ws

    def new_obs_callback(self, msg: CategoricalObservation):
        xs, ws = self.categorical_observation_to_tensors(msg)
        self.sgdrf.process_inputs(xs, ws)

    def training_step_callback(self):
        if self.sgdrf.n_xs > 0:
            loss = self.sgdrf.step()

            loss_msg = Float64()
            loss_msg.data = loss
            self.loss_publisher.publish(loss_msg)

    def point_array_to_tensor(self, point_array: List[Point]):
        coord_list = []
        if self.sgdrf.dims >= 1:
            coord_list += [[p.x for p in point_array]]
        if self.sgdrf.dims >= 2:
            coord_list += [[p.y for p in point_array]]
        if self.sgdrf.dims >= 3:
            coord_list += [[p.z for p in point_array]]
        return torch.tensor(coord_list, dtype=torch.float, device=self.sgdrf.device)

    def topic_prob_service_callback(self, request: TopicProb) -> TopicProbResponse:
        return TopicProbResponse(
            self.sgdrf.topic_prob(self.point_array_to_tensor(request.xs))
            .detach()
            .cpu()
            .squeeze()
            .tolist()
        )

    def word_prob_service_callback(self, request: WordProb) -> WordProbResponse:
        return WordProbResponse(
            self.sgdrf.word_prob(self.point_array_to_tensor(request.xs))
            .detach()
            .cpu()
            .squeeze()
            .tolist()
        )

    def word_topic_matrix_service_callback(
        self, request: WordTopicMatrix
    ) -> WordTopicMatrixResponse:
        return WordTopicMatrixResponse(torch.flatten(self.sgdrf.word_topic_prob().detach().cpu()).tolist())  # type: ignore

    def generate_parameter(self, **kwargs):
        param_name = self.param_name(kwargs["name"])
        arg_name = "--" + kwargs["name"].replace("_", "-")
        kwargs["default"] = rospy.get_param(param_name, default=kwargs["default"])
        kwargs["name"] = arg_name
        self.parser.add_argument(**kwargs)

    def setup_parameters(self):
        self.generate_parameters()
        args = vars(self.parser.parse_args())
        for k, v in args.items():
            rospy.set_param(self.param_name(k), v)

    def generate_parameters(self):
        self.generate_parameter(
            name="dims",
            value=1,
            type=int,
            description="number of spatial dimensions",
        )
        self.generate_parameter(
            name="xu_ns",
            value=25,
            type=int,
            nargs="+",
            description="number of inducing points per dimension",
        )
        self.generate_parameter(
            name="d_mins",
            value=0.0,
            type=float,
            nargs="+",
            description="minimum value of each dimension",
        )
        self.generate_parameter(
            name="d_maxs",
            value=1.0,
            type=float,
            nargs="+",
            description="maximum value of each dimension",
        )
        self.generate_parameter(
            name="V",
            value=10,
            type=int,
            description="number of observation categories",
        )
        self.generate_parameter(
            name="K",
            value=2,
            type=int,
            description="number of latent communities",
        )
        self.generate_parameter(
            name="max_obs",
            value=1000,
            type=int,
            description="maximum number of simultaneous categorical observations",
        )
        self.generate_parameter(
            name="dir_p",
            value=1.0,
            type=float,
            description="uniform dirichlet hyperparameter",
        )
        self.generate_parameter(
            name="kernel_type",
            value="Matern32",
            description="kernel type",
            choices=KernelType._member_names_,
        )
        self.generate_parameter(
            name="kernel_lengthscale",
            value=1.0,
            type=float,
            description="isotropic kernel lengthscale",
        )
        self.generate_parameter(
            name="kernel_variance",
            value=1.0,
            type=float,
            description="isotropic kernel variance",
        )
        self.generate_parameter(
            name="optimizer_type",
            value="Adam",
            description="optimizer type",
            choices=OptimizerType._member_names_,
        )
        self.generate_parameter(
            name="optimizer_lr",
            value=0.001,
            type=float,
            description="optimizer learning rate",
        )
        self.generate_parameter(
            name="optimizer_clip_norm",
            value=10.0,
            type=float,
            description="optimizer norm maximum allowed value",
        )
        self.generate_parameter(
            name="device",
            value="cpu",
            description="pytorch device to use",
        )
        self.generate_parameter(
            name="subsample_n",
            value=5,
            type=int,
            description="number of past observations for each subsample",
        )
        self.generate_parameter(
            name="subsample_type",
            value="uniform",
            description="subsample type",
            choices=SubsampleType._member_names_,
        )
        self.generate_parameter(
            name="subsample_weight",
            value=0.5,
            type=float,
            description="weight to assign to first component of compound subsample strategy",
        )
        self.generate_parameter(
            name="subsample_exp",
            value=0.1,
            type=float,
            description="exponential parameter for subsample strategy, if applicable",
        )
        self.generate_parameter(
            name="whiten",
            action="store_true",
            description="whether or not the GP inputs are whitened",
        )
        self.generate_parameter(
            name="fail_on_nan_loss",
            action="store_true",
            description="whether or not to fail if a NaN loss is encountered",
        )
        self.generate_parameter(
            name="num_particles",
            value=1,
            type=int,
            description="number of parallel samples from approximate posterior",
        )
        self.generate_parameter(
            name="jit",
            action="store_true",
            description="whether or not to JIT compile the prior and approximate posterior",
        )
        self.generate_parameter(
            name="random_seed",
            value=777,
            type=int,
            description="random seed",
        )


def main():
    node = SGDRFNode()
    node.spin()


if __name__ == "__main__":
    main()
