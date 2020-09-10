FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

# RUN apt-get update && apt-get install -y \
#     curl \
#     ca-certificates \
#     sudo \
#     git \
#     bzip2 \
#     libx11-6 \
#     && rm -rf /var/lib/apt/lists/*

RUN apt-get update

#RUN apt-get install -y --fix-missing \
#    build-essential \
#    cmake \
#    gfortran \
RUN apt-get install git \
    # wget \
    # curl \
   # graphicsmagick \
   # libgraphicsmagick1-dev \
   # libatlas-base-dev \
   # libavcodec-dev \
   # libavformat-dev \
   # libgtk2.0-dev \
   # libjpeg-dev \
   # liblapack-dev \
   # libswscale-dev \
   # pkg-config \
   # python3-dev \
    # python3-numpy \
    software-properties-common \
    # zip \
    && apt-get clean && rm -rf /tmp/* /var/tmp/*

RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt update && \
    apt install python3.6 -y && \
    apt install python3-distutils -y && \
    apt install python3.6-dev -y && \
    apt install build-essential -y && \
    apt-get install python3-pip -y && \
    apt update && apt install -y libsm6 libxext6 && \
    apt-get install -y libxrender-dev

COPY . /AI-city

RUN cd AI-city && \
    pip3 install -r requirements.txt

WORKDIR /AI-city

RUN git clone https://github.com/facebookresearch/detectron2.git && \
    python3 -m pip install -e detectron2

#ENV FORCE_CUDA="1"
# This will by default build detectron2 for all common cuda architectures and take a lot more time,
# because inside `docker build`, there is no way to tell which architecture will be used.
#ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
#ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

#RUN ./run.sh
