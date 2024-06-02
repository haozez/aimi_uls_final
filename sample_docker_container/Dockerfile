FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04 AS base

RUN rm /etc/apt/sources.list.d/cuda.list

RUN apt-get update && \
  apt-get install -y software-properties-common && \
  add-apt-repository ppa:deadsnakes/ppa && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y \
  git \
  wget \
  unzip \
  libopenblas-dev \
  python3.9 \
  python3.9-dev \
  python3-pip \
  nano \
  && \
  apt-get clean autoclean && \
  apt-get autoremove -y && \
  apt-get install ffmpeg libsm6 libxext6 -y && \
  rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3.9 -m pip install --no-cache-dir --upgrade pip
COPY requirements.txt /tmp/requirements.txt
RUN python3.9 -m pip install --no-cache-dir -r /tmp/requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

# Configure Git, clone the repository without checking out, then checkout the specific commit
RUN git config --global advice.detachedHead false && \
    git clone --no-checkout https://github.com/MIC-DKFZ/nnUNet.git /opt/algorithm/nnunet/ && \
    cd /opt/algorithm/nnunet/ && \
    git checkout 2d2e8ce2c0261dc88b53866dcc4c71e6972432ed

# Install a few dependencies that are not automatically installed
RUN pip3 install \
        -e /opt/algorithm/nnunet \
        graphviz \
        kornia \
        onnx \
        opencv-python \
        SimpleITK && \
    rm -rf ~/.cache/pip

### USER
RUN groupadd -r user && useradd -m --no-log-init -r -g user user

RUN chown -R user /opt/algorithm/

RUN mkdir -p /opt/app /input /output \
    && chown user:user /opt/app /input /output

USER user
WORKDIR /opt/app

ENV PATH="/home/user/.local/bin:${PATH}"

COPY --chown=user:user process.py /opt/app/
COPY --chown=user:user export2onnx.py /opt/app/

### ALGORITHM

# Copy custom trainers to docker
COPY --chown=user:user ./architecture/extensions/nnunetv2/ /opt/algorithm/nnunet/nnunetv2/

# Copy model checkpoint to docker
COPY --chown=user:user ./architecture/nnUNet_results/ /opt/algorithm/nnunet/nnUNet_results/

# # Copy container testing data to docker
# COPY --chown=user:user ./architecture/input/ /input/

# Set environment variable defaults
ENV nnUNet_raw="/opt/algorithm/nnunet/nnUNet_raw" \
    nnUNet_preprocessed="/opt/algorithm/nnunet/nnUNet_preprocessed" \
    nnUNet_results="/opt/algorithm/nnunet/nnUNet_results"

ENTRYPOINT [ "python3.9", "-m", "process" ]
