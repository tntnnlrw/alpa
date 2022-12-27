#!/bin/bash --login

git config --global user.name "tntnnlrw"
git config --global user.email "tntnnlrw@163.com"

sudo git config --global user.name "tntnnlrw"
sudo git config --global user.email "tntnnlrw@163.com"

sudo apt-get update
echo y|sudo apt-get install htop -y
echo y|sudo apt-get install tmux -y
echo y|sudo apt-get install psmisc -y
echo y|sudo apt-get install lsof -y
echo y|sudo apt-get install infiniband-diags -y  # ibstatus => check ib link
echo y|sudo apt-get install net-tools -y         # ifconfig
echo y|sudo apt-get install zip