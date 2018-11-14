FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
MAINTAINER MdMahmudul_Hasan@comcast.com

RUN apt-get update \
	&& apt-get install -y --no-install-recommends \
	    build-essential \
	    curl \
	    software-properties-common \
	    python3-pip \
	    python3-setuptools \
	    git \
	    unzip \
	&& add-apt-repository -y ppa:deadsnakes/ppa \
	&& apt-get update \
	&& apt-get install -y python3.6 python3.6-dev \
	&& apt-get clean \
	&& rm -rf /var/lib/apt/lists/*

RUN python3.6 -m pip install --upgrade pip && \
	python3.6 -m pip install numpy \
		tensorflow-gpu \
		dm-sonnet \
		tensorflow-probability \
		tensorflow-hub \
		imageio \
		wrapt \
		tqdm \
		pandas \
		graphviz \
		jupyterlab

RUN ln -s /usr/bin/python3.6 /usr/bin/python

WORKDIR /mnt/research-6f/mhasan/opensource/kinetics-i3d
ENTRYPOINT /bin/bash