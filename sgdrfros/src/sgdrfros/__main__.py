import rospy
import argparse
import sys
from typing import Any, Optional, Union, Tuple, List

import pyro
import pyro.util
import pyro.contrib.gp
import pyro.contrib.gp.kernels
from pyro.contrib.gp.util import conditional
import torch
import numpy as np
from geometry_msgs.msg import Point

from sgdrf_msgs.msg import CategoricalObservation
from sgdrf_srvs.srv import (
    TopicProb,
    TopicProbRequest,
    TopicProbResponse,
    WordProb,
    WordProbRequest,
    WordProbResponse,
    WordTopicMatrix,
    WordTopicMatrixRequest,
    WordTopicMatrixResponse,
    GPVariance,
    GPVarianceRequest,
    GPVarianceResponse,
)
from std_msgs.msg import Float64
from std_srvs.srv import Trigger, TriggerRequest, TriggerResponse

from sgdrf import SGDRF, SGDRFConfig, UniformSubsampler

RANGETYPE = Tuple[Union[int, float], Union[int, float], Union[int, float]]


class SGDRFNode:
    def __init__(self):
        rospy.init_node("sgdrf")
        rospy.loginfo("Starting node %s", rospy.get_name())
        self.parser = argparse.ArgumentParser()
        self.setup_parameters()
        rospy.loginfo("Successfully set up parameters")
        self.sgdrf = self.initialize_sgdrf()
        rospy.logdebug("Successfully set up sgdrf object")
        rospy.set_param("~num_words", self.sgdrf.V)
        rospy.logdebug("Using vocabulary of size %d for SGDRF model", self.sgdrf.V)
        self.obs_subscriber = rospy.Subscriber(
            f"categorical_observation__{self.sgdrf.V}__",
            CategoricalObservation,
            self.new_obs_callback,
        )
        rospy.logdebug(
            'Set up subscriber to topic "categorical_observation__%d__"', self.sgdrf.V
        )
        self.loss_publisher = rospy.Publisher("loss", Float64, queue_size=10)
        rospy.logdebug("Set up publisher for training loss")
        self.topic_prob_server = rospy.Service(
            "topic_prob", TopicProb, "topic_prob", self.topic_prob_service_callback
        )
        rospy.logdebug(
            "Set up service for topic probabilities at %s",
            self.topic_prob_server.resolved_name,
        )
        self.word_prob_server = rospy.Service(
            "word_prob", WordProb, self.word_prob_service_callback
        )
        rospy.logdebug(
            "Set up service for word probabilities at %s",
            self.word_prob_server.resolved_name,
        )
        self.word_topic_matrix_server = rospy.Service(
            "word_topic_matrix",
            WordTopicMatrix,
            self.word_topic_matrix_service_callback,
        )
        rospy.logdebug(
            "Set up service for word-topic matrix at %s",
            self.word_topic_matrix_server.resolved_name,
        )
        self.reset_sgdrf_server = rospy.Service(
            "reset_sgdrf", Trigger, self.reset_sgdrf_service_callback
        )
        rospy.logdebug(
            "Set up service for reset_sgdrf at %s",
            self.reset_sgdrf_server.resolved_name,
        )
        self.gp_variance_server = rospy.Service(
            "gp_variance",
            GPVariance,
            self.gp_variance_service_callback,
        )
        rospy.logdebug(
            "Set up service for GP variance at %s",
            self.gp_variance_server.resolved_name,
        )

        random_seed = self.param("random_seed")
        pyro.util.set_rng_seed(random_seed)
        rospy.logdebug("Set random seed to %d", random_seed)

        rospy.loginfo("Initialized SGDRFROS node.")
        self.training_timer = rospy.Timer(
            rospy.Duration(0.5), self.training_step_callback
        )
        rospy.logdebug("Set up training timer to run every .5 seconds")

    def spin(self):
        rospy.logdebug("Spinning node %s...", rospy.get_name())
        rospy.spin()
        rospy.logdebug("Spinning interrupted. Shutting down training timer...")
        self.training_timer.shutdown()
        rospy.logdebug("Training timer shut down.")

    def param_name(self, param: str):
        return "/".join([rospy.get_name(), param])

    def param(self, param: str, default: Optional[Any] = None):
        return rospy.get_param(self.param_name(param), default=default)

    def initialize_sgdrf(self):
        dims = self.param("dims")
        xu_ns = self.param("xu_ns")
        d_mins = self.param("d_mins")
        d_maxs = self.param("d_maxs")
        if isinstance(d_mins, float):
            d_mins = [d_mins]
        if isinstance(d_maxs, float):
            d_maxs = [d_maxs]
        if len(d_mins) == 1:
            d_mins = d_mins * dims
        if len(d_maxs) == 1:
            d_maxs = d_maxs * dims

        assert all(
            x < y for x, y in zip(d_mins, d_maxs)
        ), "d_mins must be smaller than d_maxs in every dimension"

        V = self.param("V")
        K = self.param("K")
        max_obs = self.param("max_obs")
        device_string = self.param("device")
        device = torch.device(device_string)
        dir_p = self.param("dir_p")
        assert dir_p > 0, "dir_p must be greater than zero"
        kernel_lengthscale = self.param("kernel_lengthscale")
        assert kernel_lengthscale > 0, "kernel_lengthscale must be greater than zero"
        kernel_variance = self.param("kernel_variance")
        assert kernel_variance > 0, "kernel_variance must be greater than zero"
        kernel = pyro.contrib.gp.kernels.Matern32(
            input_dim=dims, lengthscale=kernel_lengthscale, variance=kernel_variance
        ).to(device)
        optimizer_lr = self.param("optimizer_lr")
        assert optimizer_lr > 0, "optimizer_lr must be greater than zero"
        optimizer_clip_norm = self.param("optimizer_clip_norm")
        assert optimizer_clip_norm > 0, "optimizer_lr must be greater than zero"
        optimizer = pyro.optim.ClippedAdam(
            {"lr": optimizer_lr, "clip_norm": optimizer_clip_norm}
        )

        subsample_n = self.param("subsample_n")
        subsampler = UniformSubsampler(n=subsample_n, device=device)
        whiten = self.param("whiten")
        fail_on_nan_loss = self.param("fail_on_nan_loss")
        num_particles = self.param("num_particles")
        jit = self.param("jit")
        config = SGDRFConfig(
            xu_ns=xu_ns,
            d_mins=d_mins,
            d_maxs=d_maxs,
            V=V,
            K=K,
            max_obs=max_obs,
            dir_p=dir_p,
            kernel=kernel,
            optimizer=optimizer,
            subsampler=subsampler,
            device=device,
            whiten=whiten,
            fail_on_nan_loss=fail_on_nan_loss,
            num_particles=num_particles,
            jit=jit,
        )
        sgdrf_params = dict(
            xu_ns=xu_ns,
            d_mins=d_mins,
            d_maxs=d_maxs,
            V=V,
            K=K,
            max_obs=max_obs,
            dir_p=dir_p,
            kernel_lengthscale=kernel_lengthscale,
            kernel_variance=kernel_variance,
            optimizer_lr=optimizer_lr,
            optimizer_clip_norm=optimizer_clip_norm,
            device=device,
            subsample_n=subsample_n,
            whiten=whiten,
            fail_on_nan_loss=fail_on_nan_loss,
            num_particles=num_particles,
            jit=jit,
        )
        for param_name, param_val in sgdrf_params.items():
            rospy.logdebug("(SGDRF param) %20s: %20s", param_name, str(param_val))
        sgdrf = SGDRF(config)
        return sgdrf

    def categorical_observation_to_tensors(self, msg: CategoricalObservation):
        point = msg.point
        obs = msg.obs
        if self.sgdrf.dims == 1:
            x_raw = [point.y]
        if self.sgdrf.dims == 2:
            x_raw = [point.x, point.y]
        if self.sgdrf.dims == 3:
            x_raw = [point.x, point.y, point.z]
        xs = torch.tensor(
            np.array(x_raw), dtype=torch.float, device=self.sgdrf.device
        ).unsqueeze(0)
        ws = torch.tensor(obs, dtype=torch.int, device=self.sgdrf.device).unsqueeze(0)
        return xs, ws

    def new_obs_callback(self, msg: CategoricalObservation):
        xs, ws = self.categorical_observation_to_tensors(msg)
        self.sgdrf.process_inputs(xs, ws)

    def training_step_callback(self, event):
        if self.sgdrf.n_xs > 0:
            loss = self.sgdrf.step()
            rospy.logdebug("Training loss: %5.5f", loss)

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

    def topic_prob_service_callback(
        self, request: TopicProbRequest
    ) -> TopicProbResponse:
        rospy.logdebug("Received topic prob request.")
        return TopicProbResponse(
            self.sgdrf.topic_prob(self.point_array_to_tensor(request.xs))
            .detach()
            .cpu()
            .squeeze()
            .tolist()
        )

    def gp_variance_service_callback(
        self, request: GPVarianceRequest
    ) -> GPVarianceResponse:
        rospy.logdebug("Received GP variance request.")
        _, f_var = conditional(
            self.point_array_to_tensor(request.xs),
            self.sgdrf.xu,
            self.sgdrf.kernel,
            self.sgdrf.uloc,
            self.sgdrf.uscaletril,
            full_cov=False,
            whiten=self.sgdrf.whiten,
            jitter=self.sgdrf.jitter,
        )
        return GPVarianceResponse(f_var.detach().cpu().squeeze().tolist())

    def word_prob_service_callback(self, request: WordProbRequest) -> WordProbResponse:
        rospy.logdebug("Received word prob request.")
        return WordProbResponse(
            self.sgdrf.word_prob(self.point_array_to_tensor(request.xs))
            .detach()
            .cpu()
            .squeeze()
            .tolist()
        )

    def word_topic_matrix_service_callback(
        self, request: WordTopicMatrixRequest
    ) -> WordTopicMatrixResponse:
        rospy.logdebug("Received word-topic matrix request.")
        return WordTopicMatrixResponse(torch.flatten(self.sgdrf.word_topic_prob().detach().cpu()).tolist())  # type: ignore

    def reset_sgdrf_service_callback(self, request: TriggerRequest) -> TriggerResponse:
        ret = TriggerResponse(success=False, message="Did not run reset.")
        try:
            self.sgdrf = self.initialize_sgdrf()
            rospy.logdebug("Successfully reset sgdrf object")
            ret = TriggerResponse(success=True, message="")
        except Exception as e:
            ret = TriggerResponse(success=False, message=str(e))
        return ret

    def generate_parameter(self, **kwargs):
        param_name = self.param_name(kwargs["name"])
        arg_name = "--" + kwargs["name"].replace("_", "-")
        kwargs["default"] = rospy.get_param(
            param_name, default=kwargs.pop("value", None)
        )
        kwargs["help"] = kwargs.pop("description", "")
        kwargs.pop("name")
        self.parser.add_argument(arg_name, **kwargs)

    def setup_parameters(self):
        self.generate_parameters()
        parsed_args = self.parser.parse_args(args=rospy.myargv(argv=sys.argv)[1:])
        args = vars(parsed_args)
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
            name="whiten",
            value=False,
            action="store_true",
            description="whether or not the GP inputs are whitened",
        )
        self.generate_parameter(
            name="fail_on_nan_loss",
            value=False,
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
            value=False,
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
