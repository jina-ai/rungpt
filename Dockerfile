ARG CUDA_VERSION=11.6.0

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-setuptools python3-wheel python3-pip python3-dev git wget gcc locales \
    && apt-get clean && rm -rf /var/lib/apt/lists/*;

RUN pip3 install --upgrade pip

RUN python3 -m pip install --default-timeout=1000 --no-cache-dir torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116

# copy will almost always invalid the cache
COPY . /workspace/
WORKDIR /workspace

RUN pip3 install .

WORKDIR /workspace

ENTRYPOINT ["opengpt"]