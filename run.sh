#!/bin/bash
## change this file to your needs

echo "Adding some modules"

module add gcc-10.2


echo "#################"
echo "    COMPILING    "
echo "#################"

## dont forget to use comiler optimizations (e.g. -O3 or -Ofast)
g++ -Wall -Wextra -g -std=c++20 -O3 -march=native -funroll-loops -o network src/activation.cpp src/layer.cpp  src/loss.cpp src/main.cpp src/neural_network.cpp src/optimizers.cpp


echo "#################"
echo "     RUNNING     "
echo "#################"

## use nice to decrease priority in order to comply with aisa rules
## https://www.fi.muni.cz/tech/unix/computation.html.en
## especially if you are using multiple cores
nice -n 19 ./network
