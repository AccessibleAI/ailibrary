#!/bin/bash
sudo apt-get update
sudo apt-get -yq install pandoc
sudo DEBIAN_FRONTEND=noninteractive apt-get -yq install texlive-xetex texlive-fonts-recommended texlive-generic-recommended
