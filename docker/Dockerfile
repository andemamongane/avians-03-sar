# The avians container for the Gazebo tier: ROS 2 Jazzy with px4_msgs and AAS's
# interface messages (pinned like avians-02), the avians packages, ffmpeg and
# the docker CLI (to reach AAS's simulation container). It runs NEXT TO a
# running aerial-autonomy-stack; PX4 and Gazebo live in AAS's containers.
# The fast tier needs none of this: it runs anywhere with Python >= 3.10.
FROM ros:jazzy-ros-base
ARG PX4_MSGS_REF=86d8239e962f6939e05c3737784f60c02fa884db
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 ROS_DOMAIN_ID=1 \
    ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET AVIANS_ROOT=/workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3-pip git curl ffmpeg ros-${ROS_DISTRO}-rosbag2 ros-${ROS_DISTRO}-vision-msgs ros-${ROS_DISTRO}-geographic-msgs \
 && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /ws/src/px4_msgs && cd /ws/src/px4_msgs \
 && git init -q && git remote add origin https://github.com/PX4/px4_msgs.git \
 && git fetch --depth 1 origin ${PX4_MSGS_REF} && git checkout -q FETCH_HEAD
ARG AAS_WS=external/aerial-autonomy-stack/aircraft/aircraft_ws/src
COPY ${AAS_WS}/autopilot_interface_msgs /ws/src/autopilot_interface_msgs
RUN . /opt/ros/jazzy/setup.sh && cd /ws \
 && MAKEFLAGS=-j2 colcon build --parallel-workers 2 --cmake-args -DCMAKE_BUILD_TYPE=Release

# Python: no venv (rclpy is built for the system interpreter); numpy pinned to apt's.
RUN python3 -m pip install --break-system-packages --no-cache-dir \
      "numpy==$(python3 -c 'import numpy; print(numpy.__version__)')" \
      scipy shapely networkx pyyaml pydantic filterpy matplotlib pymavlink imageio-ffmpeg pytest ruckig "anthropic>=1"
WORKDIR /workspace
COPY packages /workspace/packages
RUN for p in env spec perception skills safety sim adp viz agent; do \
      python3 -m pip install --break-system-packages --no-cache-dir --no-deps -e /workspace/packages/avians_$p; done

ARG DOCKER_CLI=27.3.1
RUN set -eux; arch=$(dpkg --print-architecture); \
    case "$arch" in amd64) a=x86_64 ;; arm64) a=aarch64 ;; *) exit 1 ;; esac; \
    curl -fsSL "https://download.docker.com/linux/static/stable/${a}/docker-${DOCKER_CLI}.tgz" \
      | tar -xz -C /usr/local/bin --strip-components=1 docker/docker

RUN printf '%s\n' 'set +u' 'source /opt/ros/jazzy/setup.bash' 'source /ws/install/setup.bash' \
      | tee /etc/profile.d/ros.sh >> /root/.bashrc
CMD ["sleep", "infinity"]
