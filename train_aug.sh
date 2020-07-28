#!/usr/bin/env bash
nvidia-smi
cd /xh-volume/darts_ignoring
apt-get update
apt install graphviz
pip install graphviz
python augment.py --name cifar100-2e2 --dataset cifar100 --genotype "Genotype(normal=[[('skip_connect', 0), ('skip_connect', 1)], [('skip_connect', 0), ('sep_conv_3x3', 2)], [('skip_connect', 0), ('sep_conv_3x3', 1)], [('skip_connect', 0), ('sep_conv_3x3', 1)]], normal_concat=range(2, 6), reduce=[[('max_pool_3x3', 0), ('max_pool_3x3', 1)], [('skip_connect', 2), ('max_pool_3x3', 0)], [('skip_connect', 2), ('avg_pool_3x3', 0)], [('skip_connect', 2), ('max_pool_3x3', 0)]], reduce_concat=range(2, 6))"
