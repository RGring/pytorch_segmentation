#!/bin/sh
LOGPATH=$1
module load python3/3.6.7
python3 /zhome/d6/0/152995/.local/lib/python3.6/site-packages/tensorboard/main.py --logdir="$LOGPATH"