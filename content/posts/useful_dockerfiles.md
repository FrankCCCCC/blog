---
title: "Useful Dockerfile"
date: 2023-08-18T13:44:10+08:00
draft: false
author: SY Chou
weight: 1

description: "Dockerfiles for Anaconda, Jax, and non-root user"
categories: ["linux", "docker", "anaconda", "jax"]
series: []
tags: ["linux", "docker", "anaconda", "jax"]

keywords:
- Linux
- Docker
- Anaconda
- Jax

cover:
    image: "img/just_imgs/iced_land.jpg"
    relative: false # To use relative path for cover image, used in hugo Page-bundles
---

## Dockerfiles

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH $CONDAPATH/bin:$PATH

ARG USER=user
ARG PASS="0000"
ARG USERHOME="/home/${USER}"
ARG CONDAPATHI="${USERHOME}/conda"
# Add git email / username
# ARG git_email=""
# ARG git_username=""
# ARG wandb_key=""

# hadolint ignore=DL3008
RUN set -x && \
    apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
        bzip2 \
        ca-certificates \
        git \
        libglib2.0-0 \
        libsm6 \
        libxcomposite1 \
        libxcursor1 \
        libxdamage1 \
        libxext6 \
        libxfixes3 \
        libxi6 \
        libxinerama1 \
        libxrandr2 \
        libxrender1 \
        mercurial \
        openssh-client \
        procps \
        subversion \
        wget \
        tmux \ 
        htop \
        nginx \
        zip \
        nano \
        vim \
        sudo \
    && curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash \
    && curl -fsSL https://code-server.dev/install.sh | sh \
    && apt-get install git-lfs \
    && git lfs install \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -s /bin/bash $USER && echo "$USER:$PASS" | chpasswd && \
    echo "${USER} ALL=(ALL) ALL" >> /etc/sudoers

RUN UNAME_M="$(uname -m)" && \
    if [ "${UNAME_M}" = "x86_64" ]; then \
        ANACONDA_URL="https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh"; \
        SHA256SUM="e7ecbccbc197ebd7e1f211c59df2e37bc6959d081f2235d387e08c9026666acd"; \
    elif [ "${UNAME_M}" = "s390x" ]; then \
        ANACONDA_URL="https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-s390x.sh"; \
        SHA256SUM="f5ccc24aedab1f3f9cccf1945ca1061bee194fa42a212ec26425f3b77fdd943a"; \
    elif [ "${UNAME_M}" = "aarch64" ]; then \
        ANACONDA_URL="https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-aarch64.sh"; \
        SHA256SUM="fbadbfae5992a8c96af0a4621262080eea44e22baee2172e3dfb640f5cf8d22d"; \
    elif [ "${UNAME_M}" = "ppc64le" ]; then \
        ANACONDA_URL="https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-ppc64le.sh"; \
        SHA256SUM="8fdebc79f63b74daad421a2674d43299fa9c5007d85cf00e8dc1a81fbf2787e4"; \
    fi && \
    wget "${ANACONDA_URL}" -O $USERHOME/anaconda.sh -q && \
    echo "${SHA256SUM} ${USERHOME}/anaconda.sh" > $USERHOME/shasum && \
    sha256sum --check --status $USERHOME/shasum && \
    chmod -R 777 $USERHOME/shasum

USER user
RUN /bin/bash $USERHOME/anaconda.sh -b -p $CONDAPATHI

USER root
RUN rm $USERHOME/anaconda.sh $USERHOME/shasum && \
    ln -s $CONDAPATHI/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". ${CONDAPATHI}/etc/profile.d/conda.sh" >> $USERHOME/.bashrc && \
    echo "conda activate base" >> $USERHOME/.bashrc && \
    find $CONDAPATHI/ -follow -type f -name '*.a' -delete && \
    find $CONDAPATHI/ -follow -type f -name '*.js.map' -delete && \
    $CONDAPATHI/bin/conda clean -afy

# Customize the following script to install the package that you want
# SHELL ["/bin/bash", "-c"]
USER user
RUN $CONDAPATHI/bin/conda create --name py38 python=3.8 -y && \
    $CONDAPATHI/bin/conda run -n py38 pip install pyarrow==6.0.1 && \
    $CONDAPATHI/bin/conda run -n py38 pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    $CONDAPATHI/bin/conda run -n py38 pip install accelerate comet-ml matplotlib datasets tqdm tensorboard tensorboardX torchvision tensorflow-datasets einops pytorch-fid joblib PyYAML kaggle wandb torchsummary torchinfo lpips torchmetrics && \
    $CONDAPATHI/bin/conda run -n py38 pip install git+https://github.com/Database-Project-2021/scalablerunner.git
#     $CONDAPATHI/bin/conda run -n py38 wandb login --relogin --cloud $wandb_key
# RUN git config --global user.email $git_email && \
#     git config --global user.name $git_username

USER root

CMD [ "/bin/bash" ]
```

Build Image

```bash
docker build -f Dockerfile --tag=conda:1.0 .
```

Create Container

```bash
docker run -it -d --gpus all --name conda --shm-size 200G -p 5001:6006 conda:1.0
```

Attach Shell to the Container

```bash
docker exec -it conda bash
```

## Dev Container Configuration

``Ctrl+Shft+P`` and select ``Dev Containers: Open Attached Container Configuration File``or ``Dev Containers: Add Dev Container Configuration Files``. Add the following property into configuration

```json
"remoteUser": "user",
```

## Jax Docker Container

```dockerfile
FROM nvidia/cuda:11.0.3-devel-ubuntu20.04

# declare the image name
ENV IMG_NAME=11.0.3-devel-ubuntu20.04 \
    # declare what jaxlib tag to use
    # if a CI/CD system is expected to pass in these arguments
    # the dockerfile should be modified accordingly
    JAXLIB_VERSION=0.1.71

# install python3-pip
RUN apt update && apt install python3-pip -y

# install dependencies via pip
# RUN pip3 install numpy scipy six wheel jaxlib==${JAXLIB_VERSION}+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_releases.html jax[cuda11_cudnn82] -f https://storage.googleapis.com/jax-releases/jax_releases.html
RUN pip3 install numpy scipy six wheel
RUN pip3 install --upgrade "jax[cuda11_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

```