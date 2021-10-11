FROM nvcr.io/nvidia/cuda:11.4.2-cudnn8-devel-ubuntu20.04

RUN export DEBIAN_FRONTEND="noninteractive" && \
apt-get update && \
apt-get upgrade -y && \
apt-get autoremove -y && \
apt-get install --no-install-recommends -y \
cython3 \
git \
libfreetype6-dev \
pkgconf \
python3-dev \
python3-pip && \
apt-get clean -y && \
rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /requirements.txt

RUN cd / && \
python3 -m pip install --upgrade pip && \
python3 -m pip install --upgrade wheel && \
python3 -m pip install numpy==1.19.5 && \
python3 -m pip install -r requirements.txt

RUN cd / && \
git clone https://github.com/spring-epfl/CTGAN.git && \
git clone https://github.com/spring-epfl/synthetic_data_release.git && \
cd CTGAN && \
python3 -m pip install .

ENV PYTHONPATH "${PYTHONPATH}:/CTGAN"

ENTRYPOINT ["/bin/bash"]
