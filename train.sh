#!/usr/bin/env bash
nvidia-smi
cd /xh-volume/darts_ignoring
apt-get update
apt install graphviz
pip install graphviz
python search-ignore-train.py --name cifar100-no-sigmoid --dataset cifar100
