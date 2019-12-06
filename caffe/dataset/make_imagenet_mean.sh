#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

EXAMPLE=/home/edgarmg/Dropbox/Cinvestav/Clases/Redes_neuronales/github/redes_neuronales/caffe/dataset
DATA=/home/edgarmg/Dropbox/Cinvestav/Clases/Redes_neuronales/github/redes_neuronales/caffe/dataset
TOOLS=/media/edgarmg/home/Frameworks/nnetworks/caffe/build/tools

$TOOLS/compute_image_mean $EXAMPLE/train_lmdb \
  $DATA/train_mean.binaryproto

echo "Done."
