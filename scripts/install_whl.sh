apt-get update 
apt-get install -y apt-utils 
apt-get install -y python3-virtualenv 
virtualenv --python=python3.8 python3.8-env
# source python3.8-env/bin/activate && pip install --upgrade pip && pip install numpy==1.20 setuptools wheel six auditwheel

# pip install cupy-cuda111 
# pip install alpa 
# pip install --no-index --find-links=/wheels jaxlib-0.3.22+cuda111.cudnn805-cp38-cp38-manylinux2014_x86_64.whl