FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y
RUN mkdir app
WORKDIR /app

RUN python3 -m pip install --upgrade pip
RUN pip3 install \
    opencv-python \
    optuna==2.0.0 \
    pandas \
    matplotlib \
    torch \
    numpy \
    Pillow \
    bs4 \
    scikit-learn \
    torchvision \
    pytorchtools \
    joblib\
    scikit-image


RUN mkdir ./bin
COPY data_aug.py ./bin
RUN chmod 777 ./bin/data_aug.py
COPY plot_class_distribution.py ./bin
RUN chmod 777 ./bin/plot_class_distribution.py
COPY plot_images.py ./bin
RUN chmod 777 ./bin/plot_images.py
COPY rename_file.py ./bin
RUN chmod 777 ./bin/rename_file.py
COPY train_model.py ./bin
RUN chmod 777 ./bin/train_model.py
COPY hpo_train.py ./bin
RUN chmod 777 ./bin/hpo_train.py
