import torch
from ultralytics import YOLO
model = YOLO('checkpoints/yolov8n.pt')
layers = model.model
print(model.model.model[0].conv.weight.shape)  # torch.Size([64, 3, 7, 7])