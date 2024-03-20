import torch
import torch.nn as nn
import torchvision
class FasterRCNN(nn.Module):
    def __init__(self):
        super(FasterRCNN, self).__init__()
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
        model.eval()
        layers = list(model.children())
        self.to_tensor = torchvision.transforms.ToTensor()
        self.transform = layers[0]
        self.backbone = layers[1]
    def forward(self, images, targets=None):
        images = images.permute(0, 3, 1, 2)/255
        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)
        f1, f2 = features["0"], features["1"]
        features = torch.cat((f1, f2), 1)
        return features

if __name__ == "__main__":
    from datasets import load_datasets
    from config import cfg
    model = FasterRCNN()
    train_loader, test_loader = load_datasets(cfg)
    data = next(iter(train_loader))
    img1 = data["previmg"]
    out = model(img1)
    print(img1.shape)
    print(out.shape)
