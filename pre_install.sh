#!/bin/bash --login

git config --global user.name "tntnnlrw"
git config --global user.email "tntnnlrw@163.com"

sudo git config --global user.name "tntnnlrw"
sudo git config --global user.email "tntnnlrw@163.com"

sudo apt-get update
sudo apt-get install htop -y
sudo apt-get install tmux -y
sudo apt-get install psmisc -y
sudo apt-get install lsof -y
sudo apt-get install infiniband-diags -y  # ibstatus => check ib link
sudo apt-get install net-tools -y         # ifconfig