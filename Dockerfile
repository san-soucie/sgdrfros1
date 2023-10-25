FROM ros:noetic
LABEL org.opencontainers.image.source="https://github.com/san-soucie/sgdrfros1"
# Increment DEPSCACHE when there's a known change to deps.rosinstall
ARG DEPSCACHE=1
SHELL ["/usr/bin/bash", "-c"]
WORKDIR /app

RUN apt update \
 && apt-get install -y \
 build-essential \
 git \
 python3-catkin-tools \
 python3-pip \
 python3-rosdep \
 python3-vcstool \
 python3-virtualenv \
 screen \
 && rm -rf /var/lib/apt/lists/*


COPY deps.rosinstall ./

# Clone third-party dependencies from VCS
RUN echo Installing ROS dependencies:${DEPSCACHE} \
 && mkdir ./src \
 && vcs import src < deps.rosinstall

# Install dependencies declared in package.xml files
COPY sgdrfros/custom_deps.yaml ./
RUN bash -c "echo yaml file://$(readlink -f custom_deps.yaml) >> /etc/ros/rosdep/sources.list.d/20-default.list"
RUN apt update \
 && rosdep update \
 && rosdep install --default-yes --from-paths ./src --ignore-src \
 && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --upgrade transitions

# Warm the build directory with pre-built packages that don't change often.
# This list can be updated according to `catkin build --dry-run phyto_arm`.
RUN source /opt/ros/${ROS_DISTRO}/setup.bash \
 && stdbuf -o L catkin build \
        ds_core_msgs \
        ds_sensor_msgs \
        ds_util_nodes \
        rtsp_camera

# Copy the launch tool
ENV DONT_SCREEN=1
ENV NO_VIRTUALENV=1