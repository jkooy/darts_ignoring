#!/usr/bin/env bash
nvidia-smi
cd /xh-volume/darts_ignoring
apt-get update
apt install graphviz
pip install graphviz
python search.py --name cifar100-no_sigmoid --dataset cifar100
