#!/bin/sh

echo "-- Updating System"
echo " "

sudo apt update
sudo apt upgrade
sudo apt -y install python3-pip

echo "-- Installing requirements"
echo " "

sudo apt install python3-opencv
pip3 install --user matplotlib
pip3 install --user numpy
pip3 install --user scipy
pip3 install --user Pillow
sudo apt-get install python python3-tk 

echo "-- Done"