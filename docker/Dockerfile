# Base image with Python and CUDA support
FROM nvidia/cuda:11.8.0-base-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.8 python3.8-dev python3-pip \
    build-essential \
    cmake ninja-build \
    wget unzip git \
    libx11-dev libgl1-mesa-dev \
    libsdl2-dev ffmpeg x11vnc xvfb \
    retroarch libretro-core-info libretro-snes9x && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Ensure Python 3.8 is the default
RUN ln -sf /usr/bin/python3.8 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip


# Upgrade pip and setuptools
RUN pip install setuptools==58.0.4 setuptools_scm==7.1.0 pip==21

# Install Gym-Retro
RUN pip3 install git+https://github.com/Farama-Foundation/stable-retro.git

# Install other Python dependencies
RUN pip3 install stable-baselines3==1.6.2 jupyter matplotlib pandas numpy torch torchvision typer rich pytest
RUN pip3 install --no-cache-dir pre-commit==3.4.0 black==23.9.1 isort==5.12.0
RUN pip3 install pydantic opencv-python

# Create directories for ROMs, logs, and notebooks
RUN mkdir -p /home/user/roms /home/user/logs /home/user/config /home/user/notebooks
WORKDIR /home/user

# Set PATH to include local bin
ENV PATH=$PATH:/root/.local/bin

# Default command
CMD ["bash"]
