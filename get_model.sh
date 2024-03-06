#!/bin/bash

mkdir -p checkpoints

echo $1

model_size=$1

if [ $model_size == "n" ] || [ $model_size == "s" ] || [ $model_size == "m" ] || [ $model_size == "l" ] || [ $model_size == "x" ]; then
    echo "Downloading YOLOV8$model_size"
else
    echo "Invalid model size"
    exit 1
fi

cpt="https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8$model_size.pt"

echo "Downloading yolov8n.pt from $cpt"

wget -q --show-progress $cpt -O "checkpoints/yolov8$model_size.pt"
