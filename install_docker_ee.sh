#!/bin/bash --login

git config --global user.name "tntnnlrw"
git config --global user.email "tntnnlrw@163.com"

sudo git config --global user.name "tntnnlrw"
sudo git config --global user.email "tntnnlrw@163.com"

sudo apt-get remove docker docker-engine docker.io containerd runc
sudo apt-get update
echo y|sudo apt-get install \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

echo y|sudo apt-get install htop -y
echo y|sudo apt-get install tmux -y
echo y|sudo apt-get install psmisc -y
echo y|sudo apt-get install lsof -y
echo y|sudo apt-get install infiniband-diags -y  # ibstatus => check ib link
echo y|sudo apt-get install net-tools -y         # ifconfig
echo y|sudo apt-get install zip

sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
echo y|sudo apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin

sudo service docker start
sudo service docker restart

curl https://get.docker.com | sh \
  && sudo service docker restart

distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update

sudo apt-get install -y nvidia-docker2

sudo service docker restart

# change the /etc/docker/daemon.json into the next commands
#sudo cat daemon.json| /etc/docker/daemon.json
# {
#     "runtimes": {
#         "nvidia": {
#             "path": "nvidia-container-runtime",
#             "runtimeArgs": []
#         }
#     },
#     "storage-driver": "vfs"
# }
# sudo service docker restart

# bash docker_start.sh 4

