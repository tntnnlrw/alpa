# base docker image
FROM gcr.io/tensorflow-testing/nosla-cuda11.1-cudnn8-ubuntu18.04-manylinux2010-multipython

COPY dist/cuda111/jaxlib-0.3.22+cuda111.cudnn805-cp38-cp38-manylinux2014_x86_64.whl /wheels/jaxlib-0.3.22+cuda111.cudnn805-cp38-cp38-manylinux2014_x86_64.whl
# init workdir
# RUN mkdir -p /build
# WORKDIR /build

# install common tool & conda
# RUN sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
# RUN sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN apt-get update 
RUN apt-get install -y apt-utils && \
    apt-get install -y python3-virtualenv && \
    virtualenv --python=python3.8 python3.8-env && \
    source python3.8-env/bin/activate && pip install --upgrade pip && pip install numpy==1.20 setuptools wheel six auditwheel

# ARG JAX_CUDA_VERSION=11.1

# install conda alpa env
RUN pip install cupy-cuda111 && \
    pip install alpa && \
    pip3 install --no-index --find-links=/wheels jaxlib-0.3.22+cuda111.cudnn805-cp38-cp38-manylinux2014_x86_64.whl
