#!/bin/sh
#python -u train.py -g 1 -b 256 -e 1000 -o model1 --start 1
#python -u train.py -g 2 -b 256 -e 1000 -o model2 --start 1
#python -u train.py -g 3 -b 256 -e 1000 -o model3 --start 1
#python -u train.py -g 0 -b 400 -e 1500 -o model4 --start 1

python -u train.py -g 2 -b 256 -e 1500 -o model_xy -t 0 --start 1
