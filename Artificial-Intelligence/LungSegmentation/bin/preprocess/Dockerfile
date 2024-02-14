FROM ubuntu:bionic
WORKDIR /usr/bin
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa
RUN apt update && \
    apt install -y \
    python3.6 \
    python3-pip \
    libgl1-mesa-glx \
    wget \
    curl
RUN pip3 install --upgrade pip setuptools
RUN pip3 install \
    #scikit-learn \
    h5py \
    numpy==1.18.4 \
    pandas \
    opencv-python
ENV LANG C.UTF-8
