#!/bin/sh

echo "!! This program will set up everything some of the required libraries for this project !!"
echo " "

echo "-- Installing requirements"
echo " "


pip3 install --user  opencv-python==3.4.5.20
pip3 install --user JumpWayMQTT
pip3 install --user requests
pip3 install --user flask
pip3 install --user matplotlib
pip3 install --user numpy
pip3 install --user scipy
pip3 install --user Pillow
sudo apt-get install python python3-tk 

echo "-- Done"