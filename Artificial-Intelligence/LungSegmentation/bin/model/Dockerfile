FROM ryantanaka/cuda10.1:latest

WORKDIR /usr/bin
RUN apt update && \
    apt install -y \
    libgl1-mesa-glx

RUN pip3 install --upgrade pip setuptools
RUN pip3 install \
    tensorflow==2.1.0 \
    h5py \
    numpy==1.18.4 \
    pandas \
    opencv-python \
    keras==2.3.1 \
    optuna \
    segmentation_models \
    dataclasses \
    matplotlib \
    reportlab \
    joblib