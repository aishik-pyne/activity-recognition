#!/usr/bin/env bash

echo "Downloading config files..."

mkdir -p /data/networks/yolov3/cfg
wget -O /data/networks/yolov3/cfg/coco.data https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/coco.data
wget -O /data/networks/yolov3/cfg/yolov3.cfg https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg

mkdir -p /data/networks/yolov3/data
wget -O /data/networks/yolov3/data/coco.names https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names

echo "Downloading yolov3 weights"
mkdir -p /data/networks/yolov3/weights
wget -O /data/networks/yolov3/weights/yolov3.weights https://pjreddie.com/media/files/yolov3.weights