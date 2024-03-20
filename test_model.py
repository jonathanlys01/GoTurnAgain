import torch
from ultralytics import YOLO
model = YOLO('checkpoints/yolov8n.pt')
print(model.task)
print(list(model.model.model)[2])