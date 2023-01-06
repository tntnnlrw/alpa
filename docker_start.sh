#!/bin/bash --login

a=$1

# sudo docker run --gpus ${a} -v $(pwd):/build --rm --shm-size=10.24gb -it tntnn/alpa:0.1
sudo docker run --gpus ${a} -v $(pwd):/build --rm --shm-size=10.24gb -it gcr.io/tensorflow-testing/nosla-cuda11.1-cudnn8-ubuntu18.04-manylinux2010-multipython
