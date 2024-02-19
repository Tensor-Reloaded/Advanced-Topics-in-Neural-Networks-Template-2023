FROM nvidia/cuda:12.3.1-base-ubuntu22.04

ENV DEVICE cpu
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y \
    software-properties-common \
    build-essential \
    wget \
    zlib1g-dev \
    libssl-dev \
    libffi-dev \
    openssl \
    libz-dev \
    libbz2-dev \
    liblzma-dev \
    libreadline-dev \
    libncursesw5-dev \
    libgdbm-dev \
    libsqlite3-dev \
    libc6-dev \
    libopenblas-dev \
    liblapack-dev \
    libblas-dev \
    cmake \
    unzip \
    time

# Download and install Python 3.11.0
WORKDIR /usr/src/app

RUN wget https://www.python.org/ftp/python/3.11.0/Python-3.11.0.tgz && \
    tar -xzf Python-3.11.0.tgz && \
    rm Python-3.11.0.tgz && \
    cd Python-3.11.0 && \
    ./configure --enable-optimizations && \
    make altinstall


RUN python3.11 -m pip install torch torchvision onnxruntime psutil

WORKDIR /usr/src/libtorch_cpu

# Download libtorch
RUN wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.2.0%2Bcpu.zip  && \
    unzip libtorch-cxx11-abi-shared-with-deps-2.2.0+cpu.zip && \
    rm libtorch-cxx11-abi-shared-with-deps-2.2.0+cpu.zip

WORKDIR /usr/src/app

COPY . .

# Build the C++ code
WORKDIR /usr/src/app/cpp/example-app

RUN mv /usr/src/libtorch_cpu/libtorch /usr/src/app/cpp/example-app/libtorch && \
    rm -rf build/* && \
    cmake -S . -B build && cmake --build build --config Release

WORKDIR /usr/src/app

ENTRYPOINT /bin/bash driver.sh
