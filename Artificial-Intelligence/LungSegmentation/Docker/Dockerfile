FROM opensciencegrid/tensorflow-gpu:2.3-cuda-10.1
WORKDIR /usr/bin
RUN apt-get update && \
    apt-get install -y software-properties-common wget curl openssh-client && \
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
    scikit-learn \
    tensorflow==2.1.0\
    h5py \
    numpy==1.18.4 \
    pandas \
    opencv-python \
    keras==2.3.1 \
    optuna \
    segmentation_models\
    dataclasses \
    tabulate  \
    matplotlib \
    reportlab  \
    joblib
ENV LANG C.UTF-8
