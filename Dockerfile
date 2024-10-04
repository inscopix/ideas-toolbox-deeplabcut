FROM nvidia/cuda:11.2.2-devel-ubuntu20.04 AS base

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONFAULTHANDLER=1
ENV DEBIAN_FRONTEND=noninteractive
# disable console output of tqdm since it can cause issues in the cloud
ENV TQDM_DISABLE=1

# adding this to path so that we can source python correctly. 
# this is where the python version we install using apt will
# live
ENV PATH="${PATH}:/ideas/.local/bin"

ARG PACKAGE_REQS

ARG PYTHON_VERSION=3.10.0
ARG PYTHON=python3.10

ARG OPENCV_VERSION=4.6.0

ENV PACKAGE_REQS=$PACKAGE_REQS

# Create ideas user
RUN addgroup ideas \
    && adduser --disabled-password --home /ideas --ingroup ideas ideas

WORKDIR /ideas

RUN apt update && apt upgrade -y \
    && apt install -y software-properties-common \
    && apt install -y gcc python3-dev \
    && apt install -y libgl1-mesa-glx libglib2.0-0 \
    && apt install -y ffmpeg liblzma-dev

# BE needs at least python3.9, but system python on ubuntu 20.04 is python3.8
# build python3.10 from source, instead of installing through apt because
# we need to build opencv from source to get h264 support (not available on opencv-python pip package),
# and the python bindings for opencv were not building when python3.10 was installed from apt.
RUN apt install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libsqlite3-dev libreadline-dev libffi-dev wget libbz2-dev curl \
    && mkdir -p $HOME/opt \
    && cd $HOME/opt \
    && curl -O https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz \
    && tar -xzf Python-${PYTHON_VERSION}.tgz \
    && cd Python-${PYTHON_VERSION} \
    && ./configure --enable-shared --enable-optimizations --prefix=/usr/local LDFLAGS="-Wl,--rpath=/usr/local/lib" \
    && make altinstall

RUN apt install -y git python3-pip \
    && ${PYTHON} -m pip install --upgrade pip \
    && ${PYTHON} -m pip install --no-cache-dir awscli boto3 click requests    

# link python to the version of python BE needs
RUN ln -s $(which ${PYTHON}) /usr/bin/python

# copy code and things we need
COPY setup.py function_caller.py user_deps.txt ./

# install dependencies
RUN ${PYTHON} -m pip install --default-timeout=1000 -e .

# for training, dlc is for some reason writing to files here
RUN chown -R ideas:ideas /usr/local/lib/${PYTHON}/site-packages/deeplabcut

ARG OPENCV_VERSION=4.6.0
# build opencv from source in order to have access to h264 encoder, which
# is not available in the opencv-python pip package due to licensing restrictions
# h264 encoder is required in order for movies to be playble in the browser.
RUN ${PYTHON} -m pip uninstall -y opencv-python \
    && apt-get install -y build-essential cmake pkg-config unzip yasm git checkinstall wget \
    libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev libavresample-dev \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    libxvidcore-dev x264 libx264-dev libfaac-dev libmp3lame-dev libtheora-dev  \
    libfaac-dev libmp3lame-dev libvorbis-dev \
    libatlas-base-dev gfortran \
    && cd /tmp \
    && wget -O opencv.zip https://github.com/opencv/opencv/archive/refs/tags/${OPENCV_VERSION}.zip \
    && wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/refs/tags/${OPENCV_VERSION}.zip \
    && unzip opencv.zip \
    && unzip opencv_contrib.zip \
    && cd opencv-${OPENCV_VERSION} \
    && mkdir build \
    && cd build \
    && cmake \
    -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=$(${PYTHON} -c "import sys; print(sys.prefix)") \
    -D INSTALL_PYTHON_EXAMPLES=OFF \
    -D INSTALL_C_EXAMPLES=OFF \
    -D WITH_TBB=ON \
    -D WITH_V4L=ON \
    -D WITH_OPENGL=ON \
    -D WITH_GSTREAMER=ON \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D OPENCV_PC_FILE_NAME=opencv.pc \
    -D OPENCV_ENABLE_NONFREE=ON \
    -D OPENCV_PYTHON3_INSTALL_PATH=$(${PYTHON} -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
    -D OPENCV_EXTRA_MODULES_PATH=/tmp/opencv_contrib-${OPENCV_VERSION}/modules \
    -D BUILD_NEW_PYTHON_SUPPORT=ON \
    -D BUILD_opencv_python3=ON \
    -D HAVE_opencv_python3=ON \
    -D PYTHON3_EXECUTABLE=$(which ${PYTHON}) \
    -D PYTHON_DEFAULT_EXECUTABLE=$(which ${PYTHON}) \
    -D PYTHON_INCLUDE_DIRS=$(${PYTHON} -c "from sysconfig import get_paths as gp; print(gp()['include'])") \
    -D PYTHON_LIBRARY=/usr/local/lib/lib${PYTHON}.so \
    -D BUILD_EXAMPLES=OFF .. \
    && make -j4 \
    && make install \
    && sh -c 'echo "/usr/local/lib" >> /etc/ld.so.conf.d/opencv.conf' \
    && ldconfig \
    && rm -rf /tmp/opencv-${OPENCV_VERSION} \
    && rm -rf /tmp/opencv_contrib-${OPENCV_VERSION} \
    && rm /tmp/opencv.zip \
    && rm /tmp/opencv_contrib.zip

# Clean up apt cache
RUN apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && apt autoremove

# copy models to image
COPY --chown=ideas resources/models /ideas/models
    
COPY --chown=ideas toolbox /ideas/toolbox

# Copy tool commands to image after installing the code
# so that the image is not rebuilt if a command is updated.
COPY --chown=ideas commands /ideas/commands

# Marks commands as executable.
# Return 0 always in the case there are no tool commands.
RUN chmod +x /ideas/commands/* ; return 0

# copy JSON files in info
# this includes the toolbox_info.json, and annotation files
# that are used to generate output manifests
COPY --chown=ideas info /ideas/info

# dlc requires list of videos when creating a new project,
# but they're not used at all after labeled data is generated.
# just provide any mp4 movies in data dir.
COPY --chown=ideas data/2023-01-27-10-34-22-camera-1_trimmed_1s_dlc_labeled_movie.mp4 /ideas/data/

USER ideas
CMD ["/bin/bash"]
