#!/usr/bin/env bash
nvidia-smi
cd /xh-volume/darts_ignoring
apt-get update
apt install graphviz
pip install graphviz
python search-ignore-train-sigmoid.py --name cifar100-sigmoid --dataset cifar100 --batch_size 16
