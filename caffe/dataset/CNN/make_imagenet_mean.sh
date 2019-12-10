#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

EXAMPLE=/home/edgar/Caffe_course/example2/dataset
DATA=/home/edgar/Caffe_course/example2/dataset
TOOLS=/home/edgar/caffe/build/tools

$TOOLS/compute_image_mean $EXAMPLE/train_lmdb \
  $DATA/train_mean.binaryproto

echo "Done."
