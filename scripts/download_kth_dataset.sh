#!/usr/bin/env bash

mkdir -p /data/datasets/kth_dataset 

echo "Downloading walking.zip..."
wget -P /data/datasets/kth_dataset/ http://www.nada.kth.se/cvap/actions/walking.zip
unzip /data/datasets/kth_dataset/walking.zip && rm /data/datasets/kth_dataset/walking.zip

echo "Downloading jogging.zip..."
wget -P /data/datasets/kth_dataset/ http://www.nada.kth.se/cvap/actions/jogging.zip
unzip /data/datasets/kth_dataset/jogging.zip && rm /data/datasets/kth_dataset/jogging.zip

echo "Downloading running.zip..."
wget -P /data/datasets/kth_dataset/ http://www.nada.kth.se/cvap/actions/running.zip
unzip /data/datasets/kth_dataset/running.zip && rm /data/datasets/kth_dataset/running.zip

echo "Downloading boxing.zip..."
wget -P /data/datasets/kth_dataset/ http://www.nada.kth.se/cvap/actions/boxing.zip
unzip /data/datasets/kth_dataset/boxing.zip && rm /data/datasets/kth_dataset/boxing.zip

echo "Downloading handwaving.zip..."
wget -P /data/datasets/kth_dataset/ http://www.nada.kth.se/cvap/actions/handwaving.zip
unzip /data/datasets/kth_dataset/handwaving.zip && rm /data/datasets/kth_dataset/handwaving.zip

echo "Downloading handclapping.zip..."
wget -P /data/datasets/kth_dataset/ http://www.nada.kth.se/cvap/actions/handclapping.zip
unzip /data/datasets/kth_dataset/handclapping.zip && rm /data/datasets/kth_dataset/handclapping.zip