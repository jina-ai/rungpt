ARG CUDA_VERSION=11.7.0

# devel needed for bitsandbytes requirement of libcudart.so, otherwise runtime sufficient
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends\
    git \
    curl \
    wget \
    gcc \
    locales \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt install -y python3.10 libpython3.10-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 0

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3

RUN python3 -m pip install --default-timeout=1000 --no-cache-dir torch torchvision

# copy will almost always invalid the cache
COPY . /workspace/
WORKDIR /workspace

RUN python3.10 -m pip install -e .

WORKDIR /workspace

ENTRYPOINT ["opengpt"]