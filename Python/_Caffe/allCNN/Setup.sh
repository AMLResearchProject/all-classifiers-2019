#!/bin/sh

echo "!! This program will set up everything some of the required libraries for this project !!"
echo " "

echo "-- Upgrading pip3"
echo " "

pip3 install --user --upgrade pip

echo "-- Installing requirements"
echo " "

pip3 install --user scikit-image
pip3 install --user lmdb
pip3 install --user pydot
sudo apt-get install graphviz libgraphviz-dev
pip3 install --user pygraphviz

echo "-- Done" 