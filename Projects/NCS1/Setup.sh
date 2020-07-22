#!/bin/sh

echo "-- Installing requirements"
echo " "

pip3 install --user requests
pip3 install --user flask
pip3 install --user matplotlib
pip3 install --user numpy
pip3 install --user scipy
pip3 install --user scikit-image
pip3 install --user Pillow
pip3 install --user jsonpickle
sudo apt-get install python3-tk 

echo "-- Done"