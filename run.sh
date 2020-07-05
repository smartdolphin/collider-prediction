#!/bin/sh
#python -u train.py -g 0 -b 256 -e 500 -o model1 --start 0
#python -u train.py -g 1 -b 256 -e 500 -o model2 --start 0
#python -u train.py -g 3 -b 256 -e 500 -o model4 --start 0

#python -u train.py -g 2 -b 512 -e 500 -o model_xy -t 0
#python -u predict.py

#python -u train.py -g 1 -n inception -b 64 -e 1500 -o inception1
#python -u train.py -g 2 -n inception -b 64 -e 1500 -o inception2



#python -u train.py -g 0 -t 0 -n inception -b 128 -e 1500 -o inception_xy
#python -u train.py -g 3 -t 0 -n inceptionv2 -b 128 -e 1500 -o inception_xy_maxpool


#python -u train.py -g 0 -n inceptionv2 -b 100 -e 1500 -o inception_mel0
#python -u train.py -g 1 -n inceptionv2 -b 128 -e 1500 -o inception_mel1
#python -u train.py -g 2 -n inceptionv2 -b 100 -e 1500 -o inception_mel2
python -u train.py -g 3 -n inceptionv2 -b 64 -e 1500 -o inception_mel3
