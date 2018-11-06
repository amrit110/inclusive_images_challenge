FROM nvidia/cuda:9.0-devel-ubuntu16.04

# To build, `docker build -f Dockerfile <repo_root>`

ENV CUDNN_VERSION=7.1.4.18

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        apt-utils \
        build-essential \
        curl \
        ca-certificates \
        dialog \
        gdb \
        git \
        libcudnn7=${CUDNN_VERSION}-1+cuda9.0 \
        libcudnn7-dev=${CUDNN_VERSION}-1+cuda9.0 \
        libgtk2.0-dev \
        libjpeg-dev \
        libopenexr-dev \
        libpng-dev \
        screen \
        ssh \
        vim \
        wget && \
    rm -rf /var/lib/apt/lists/*

# Python support
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-dev\
        python3-setuptools \
        python3-tk && \
    rm -rf /var/lib/apt/lists/*

# Set python default to python3
RUN ln -sfn /usr/bin/python3 /usr/bin/python

# Install pip.
# Must be done through the Python script instead of Ubuntu's package manager,
# in order to be able to upgrade it later.
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

# Upgrade pip
RUN pip install --upgrade pip
RUN pip install wheel

# Install deps
COPY requirements.txt .
RUN pip install -r requirements.txt
