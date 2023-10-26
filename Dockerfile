FROM ros:noetic AS base
LABEL org.opencontainers.image.source="https://github.com/san-soucie/sgdrfros1"
# Increment DEPSCACHE when there's a known change to deps.rosinstall
ARG DEPSCACHE=1
SHELL ["/usr/bin/bash", "-c"]
WORKDIR /app



RUN apt update \
        && apt-get install -y \
        python3-pip \
        python3-rosdep \
        screen \
        git \
        && rm -rf /var/lib/apt/lists/* \
        && python3 -m pip install --upgrade pip setuptools

COPY --link ./sgdrfros/custom_deps.yaml /custom_deps.yaml
RUN bash -c "echo yaml file://$(readlink -f /custom_deps.yaml) >> /etc/ros/rosdep/sources.list.d/20-default.list"

FROM base AS builder

RUN apt update \
        && apt-get install -y \
        build-essential \
        python3-catkin-tools \
        python3-vcstool \
        python3-virtualenv \
        && rm -rf /var/lib/apt/lists/*

COPY deps.rosinstall ./

# Clone third-party dependencies from VCS
RUN echo Installing ROS dependencies:${DEPSCACHE} \
        && mkdir ./src \
        && vcs import src < deps.rosinstall

RUN catkin config --merge-install --merge-devel --install

# Install dependencies declared in package.xml files
RUN apt update \
        && rosdep update \
        && rosdep install --default-yes --from-paths ./src --ignore-src --skip-keys=sgdrf_msgs \
        && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install -r /app/src/PhytO-ARM/deps/python3-requirements.txt

RUN sed -i 's/"Phins"/"PhinsConfig"/' /app/src/ds_msgs/ds_sensor_msgs/cfg/PhinsConfig.cfg
# Warm the build directory with pre-built packages that don't change often.
# This list can be updated according to `catkin build --dry-run phyto_arm`.
RUN source /opt/ros/${ROS_DISTRO}/setup.bash \
        && stdbuf -o L catkin build \
        ds_core_msgs \
        ds_sensor_msgs \
        ds_util_nodes \
        rtsp_camera \
        ds_sensor_parsers \
        ds_sensors        \
        ds_util_nodes     \
        ds_acomms_msgs    \
        ds_actuator_msgs  \
        ds_asio           \
        ds_hotel_msgs     \
        ds_multibeam_msgs \
        ds_mx_msgs        \
        ds_nav_msgs       \
        ds_control_msgs   \
        ds_nmea_msgs      \
        ds_nmea_parsers   \
        ds_ocomms_msgs    \
        ds_param          \
        ds_base

RUN source /opt/ros/${ROS_DISTRO}/setup.bash \
        && stdbuf -o L catkin build \
        aml_ctd           \
        ifcb              \
        jvl_motor         \
        phyto_arm         \
        rbr_maestro3_ctd 

RUN mkdir -p /app/src/sgdrfros1/sgdrf_controller \
        && mkdir /app/src/sgdrfros1/sgdrf_msgs \
        && mkdir /app/src/sgdrfros1/sgdrf_srvs \
        && mkdir /app/src/sgdrfros1/sgdrfros
COPY --link ./sgdrf_controller /app/src/sgdrfros1/sgdrf_controller
COPY --link ./sgdrf_msgs /app/src/sgdrfros1/sgdrf_msgs
COPY --link ./sgdrf_srvs /app/src/sgdrfros1/sgdrf_srvs
COPY --link ./sgdrfros /app/src/sgdrfros1/sgdrfros

RUN apt update \
        && rosdep update \
        && rosdep install --default-yes --from-paths ./src --ignore-src -t build \
        && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --upgrade transitions

RUN source /opt/ros/${ROS_DISTRO}/setup.bash \
        && stdbuf -o L catkin build

FROM base

COPY --from=builder /app/install /app/install
COPY --link --from=builder /app/src/PhytO-ARM/deps/python3-requirements.txt /phyto-arm_python_reqs.txt
RUN apt update \
        && rosdep update \
        && rosdep install -t exec --default-yes --from-paths /app/install/share --ignore-src \
        && rm -rf /var/lib/apt/lists/* \
        && python3 -m pip install --upgrade transitions \
        && python3 -m pip install -r /phyto-arm_python_reqs.txt
RUN sed -i 's|source|a source /app/install/local_setup.bash|' /ros_entrypoint.sh
# Copy the launch tool
ENV DONT_SCREEN=1
ENV NO_VIRTUALENV=1