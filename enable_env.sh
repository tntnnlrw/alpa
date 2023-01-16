#!/bin/bash --login

#conda init bash

# conda activate alpa 
# python3.8-env/bin/activate
virtualenv --python=python3.8 python3.8-env
source python3.8-env/bin/activate
echo y|ray start --head 
cd benchmark/alpa
