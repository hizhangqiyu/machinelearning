Note
------------------------
    Base on Ubuntu.


nvidia-docker
-------------------------
## Prerequisites
    1. GNU/Linux x86_64 with kernel version > 3.10
    2. Docker >= 1.9 (official docker-engine only)
    3. NVIDIA GPU with Architecture > Fermi (2.1)
    4. NVIDIA drivers >= 340.29 with binary nvidia-modprobe

### Install Nvidia Drivers
    download *.run file from http://www.nvidia.com/object/unix.html
    execute this file.

    sudo apt-get install nvidia-current-updates

## Install Nvidia Docker
    wget -P /tmp https://github.com/NVIDIA/nvidia-docker/releases/download/v1.0.1/nvidia-docker_1.0.1-1_amd64.deb
    sudo dpkg -i /tmp/nvidia-docker*.deb && rm /tmp/nvidia-docker*.deb

## Test
    nvidia-docker run --rm nvidia/cuda nvidia-smi

Command
-----------------------------

## docker
    docker run -it <docker image> <command>
    docker images
    docker build -t <image> .
    docker login
    docker tag <source image> <dest image>:<tag>
