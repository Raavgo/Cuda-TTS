# TTS model
## Authors
* Chory Matthias
* Dickbauer Alexander
* Langer Antonia

## Overview
This project is part of our academic course, it consists of a docker container that trains a text-to-speech model using part of the M-AILABS Speech Dataset.

## Setup
In order to train the model at a decent speed GPU support needs to be enabled. This is achieved by downloading and enabeling the NVIDIA Container Toolkit.

### Steps
1. Setup a Debian Based OS, preferable Ubuntu 18.04 or 20.04. 
2. Download and Install Ubuntu Nvidia Drivers (Restart after!)
3. Run setup.sh with root privileges

## How to train the model
1. Change the parameter in train.py
2. Run docker-compose build
3. Run docker-compose up or docker-compose up -d
