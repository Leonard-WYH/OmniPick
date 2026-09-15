ARG BASE_IMAGE=nvcr.io/nvidia/cuda:12.8.0-runtime-ubuntu22.04
FROM ${BASE_IMAGE}

ARG TORCH_VERSION=2.7.1+cu128
ARG TORCHVISION_VERSION=0.22.1+cu128

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV ROS_DISTRO=humble
ENV ROS_DOMAIN_ID=99
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/workspace/baseline/src

SHELL ["/bin/bash", "-c"]

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      gnupg \
      lsb-release \
      software-properties-common \
    && add-apt-repository universe \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo ${UBUNTU_CODENAME}) main" \
      > /etc/apt/sources.list.d/ros2.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
      libgl1 \
      libglib2.0-0 \
      python3-pip \
      python3-setuptools \
      python3-wheel \
      python-is-python3 \
      ros-humble-ros-base \
      ros-humble-ros2topic \
      ros-humble-cv-bridge \
      ros-humble-tf2-ros \
      ros-humble-geometry-msgs \
      ros-humble-nav-msgs \
      ros-humble-sensor-msgs \
      ros-humble-std-msgs \
      ros-humble-vision-msgs \
      ros-humble-rmw-cyclonedds-cpp \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python3 -m pip install --no-cache-dir \
      "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
      --index-url https://download.pytorch.org/whl/cu128 \
    && python3 -m pip install --no-cache-dir \
      "numpy==1.26.4" \
      "scipy==1.11.4" \
      "mujoco==3.2.7" \
      "opencv-contrib-python-headless==4.10.0.84" \
      "ultralytics==8.0.196"

COPY entrypoint.sh /usr/local/bin/supermarket-client-entrypoint
RUN chmod +x /usr/local/bin/supermarket-client-entrypoint \
    && mkdir -p /workspace/baseline

WORKDIR /workspace/baseline
ENTRYPOINT ["/usr/local/bin/supermarket-client-entrypoint"]
CMD ["sleep", "infinity"]
