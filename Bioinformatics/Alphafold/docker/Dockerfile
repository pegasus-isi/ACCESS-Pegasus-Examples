# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG CUDA_FULL=11.2.2
FROM nvidia/cuda:${CUDA_FULL}-base-ubuntu20.04 AS build
# FROM directive resets ARGS, so we specify again (the value is retained if
# previously set).
ARG CUDA_FULL
ARG CUDA=11.2
ARG DEBIAN_FRONTEND=noninteractive

# Use bash to support string substitution.
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
      build-essential \
      cmake \
      cuda-command-line-tools-${CUDA/./-} \
      git \
      hmmer \
      kalign \
      tzdata \
      wget \
    && rm -rf /var/lib/apt/lists/*

# Compile HHsuite from source.
RUN git clone --branch v3.3.0 https://github.com/soedinglab/hh-suite.git /tmp/hh-suite \
    && mkdir /tmp/hh-suite/build \
    && pushd /tmp/hh-suite/build \
    && cmake -DCMAKE_INSTALL_PREFIX=/opt/hhsuite .. \
    && make -j 4 && make install \
    && ln -s /opt/hhsuite/bin/* /usr/bin \
    && popd \
    && rm -rf /tmp/hh-suite

# Install Miniconda package manager.
RUN wget -q -P /tmp \
  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash /tmp/Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda \
    && rm /tmp/Miniconda3-latest-Linux-x86_64.sh

# Install conda packages.
ENV PATH="/opt/conda/bin:$PATH"
RUN conda install -qy conda==4.13.0 \
    && conda install -y -c conda-forge \
      openmm=7.5.1 \
      python=3.8 \
      cudatoolkit==${CUDA_FULL} \
      pdbfixer \
      pip \
    && /opt/conda/bin/conda clean -aqy

COPY . /app/alphafold
RUN wget -q -P /app/alphafold/alphafold/common/ \
  https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt

# Install pip packages.
RUN pip3 install --upgrade pip --no-cache-dir \
    && pip3 install -r /app/alphafold/requirements.txt --no-cache-dir \
    && pip3 install --upgrade --no-cache-dir \
      jax==0.3.17 \
      jaxlib==0.3.15+cuda11.cudnn805 \
      -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Apply OpenMM patch.
WORKDIR /opt/conda/lib/python3.8/site-packages
RUN patch -p0 < /app/alphafold/docker/openmm.patch

FROM nvidia/cuda:${CUDA_FULL}-base-ubuntu20.04
ARG OS=ubuntu2004
ARG CUDA=11.2

SHELL ["/bin/bash", "-c"]

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get upgrade -y \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      software-properties-common \
      cuda-command-line-tools-${CUDA/./-} \
      hmmer \
      kalign \
      wget \
    && rm -rf /var/lib/apt/lists/*

# Install cuDNN and cuSOLVER
RUN wget -q https://developer.download.nvidia.com/compute/cuda/repos/${OS}/x86_64/cuda-${OS}.pin \
    -O /etc/apt/preferences.d/cuda-repository-pin-600 \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/${OS}/x86_64/7fa2af80.pub \
    && add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/${OS}/x86_64/ /" \
    && apt-get update \
    && apt-get install libcudnn8 libcusolver-${CUDA/./-} \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /app /app
COPY --from=build /opt/conda /opt/conda
COPY --from=build /opt/hhsuite /opt/hhsuite

# hhsuite executable path hardcoded as /usr/bin in run_alphafold.py
RUN ln -s /opt/hhsuite/bin/* /usr/bin
# Add SETUID bit to the ldconfig binary so that non-root users can run it.
RUN chmod u+s /sbin/ldconfig.real

# We need to run `ldconfig` first to ensure GPUs are visible, due to some quirk
# with Debian. See https://github.com/NVIDIA/nvidia-docker/issues/1399 for
# details.
# ENTRYPOINT does not support easily running multiple commands, so instead we
# write a shell script to wrap them up.

WORKDIR /app/alphafold
RUN echo $'#!/bin/bash\n\
ldconfig\n\
python /app/alphafold/run_alphafold.py "$@"' > /app/run_alphafold.sh \
  && chmod +x /app/run_alphafold.sh

ENV PATH=/opt/conda/bin:$PATH \
    LC_ALL=C.UTF-8 \
    ALPHAFOLD_BASE_DIR=/app/alphafold
    
ENV PYTHONPATH "${PYTHONPATH}:/usr/src/app/alphafold"

ENTRYPOINT ["/app/run_alphafold.sh"]
