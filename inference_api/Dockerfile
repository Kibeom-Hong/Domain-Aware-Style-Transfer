FROM continuumio/miniconda3 AS build

COPY environment.yml .
RUN conda env create -f environment.yml

RUN conda install -c conda-forge conda-pack

RUN conda-pack -n project-matrix -o /tmp/env.tar && \
  mkdir /venv && cd /venv && tar xf /tmp/env.tar && \
  rm /tmp/env.tar

RUN /venv/bin/conda-unpack

FROM nvidia/cuda:9.0-base-ubuntu16.04 AS runtime

COPY --from=build /venv /venv

RUN apt-get update
RUN apt-get -y upgrade
RUN apt-get -y install --no-install-recommends syslog-ng

RUN apt-get update && apt-get install -y \
  libatlas-base-dev \
  libopenjp2-7 \
  libtiff-tools \
  i2c-tools \
  libsm6 \
  libxext6 \
  libwebp-dev \
  libpng16-16 \
  libjpeg-dev \
  libpng-dev \
  libtiff-dev \
  libavcodec-dev \
  libavformat-dev \
  libswscale-dev \
  libv4l-dev \
  libxvidcore-dev \
  libx264-dev \
  libgtk-3-dev \
  libatlas-base-dev \
  gfortran \
  libilmbase-dev \
  libopenexr-dev \
  libgstreamer1.0-dev \
  libqtgui4 \
  libqt4-test \
  libgl1

RUN rm -rf /var/lib/apt/lists/* \
  && apt-get -y autoremove

COPY ./src /app

WORKDIR /app

SHELL ["/bin/bash", "-c"]

ENTRYPOINT source /venv/bin/activate project-matrix && python app.py