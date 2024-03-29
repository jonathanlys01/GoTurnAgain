import torch
from torchvision import models
import torch.nn as nn
from fasterrcnn import FasterRCNN


class GoNet(nn.Module):
    """ Neural Network class
        Two stream model:
        ________
       |        | conv layers              Untrained Fully
       |Previous|------------------>|      Connected Layers
       | frame  |                   |    ___     ___     ___
       |________|                   |   |   |   |   |   |   |   fc4
                   Pretrained       |   |   |   |   |   |   |    * (left)
                   CaffeNet         |-->|fc1|-->|fc2|-->|fc3|--> * (top)
                   Convolution      |   |   |   |   |   |   |    * (right)
                   layers           |   |___|   |___|   |___|    * (bottom)
        ________                    |   (4096)  (4096)  (4096)  (4)
       |        |                   |
       | Current|------------------>|
       | frame  |
       |________|

    """
    def __init__(self):
        super(GoNet, self).__init__()
        caffenet = models.alexnet(pretrained=True)
        self.convnet = nn.Sequential(*list(caffenet.children())[:-1])
        for param in self.convnet.parameters():
            param.requires_grad = False
        self.classifier = nn.Sequential(
                nn.Linear(256*6*6*2, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4),
                )
        self.weight_init()

    def weight_init(self):
        for m in self.classifier.modules():
            # fully connected layers are weight initialized with
            # mean=0 and std=0.005 (in tracker.prototxt) and
            # biases are set to 1
            # tracker.prototxt link: https://goo.gl/iHGKT5
            if isinstance(m, nn.Linear):
                m.bias.data.fill_(1)
                m.weight.data.normal_(0, 0.005)

    def forward(self, x, y):
        x1 = self.convnet(x)
        x1 = x1.view(x.size(0), 256*6*6)
        x2 = self.convnet(y)
        x2 = x2.view(x.size(0), 256*6*6)
        x = torch.cat((x1, x2), 1)
        x = self.classifier(x)
        return x

def get_fasterrccnn_mobilenet_backbone():
    model = models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
    layers = list(model.children())
    backbone = torch.nn.Sequential(*layers[:2])
    return backbone

class FasterGTA(nn.Module):
    def __init__(self):
        super(FasterGTA, self).__init__()
        self.backbone = FasterRCNN()
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.classifier = nn.Sequential(
                nn.Linear(512*10*10*2, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4),
                )
        self.weight_init()
    def weight_init(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                m.bias.data.fill_(1)
                m.weight.data.normal_(0, 0.005)
    def forward(self, x, y):
        x1 = self.backbone(x)
        x1 = x1.view(x.size(0), 512*10*10)
        x2 = self.backbone(y)
        x2 = x2.view(x.size(0), 512*10*10)
        x = torch.cat((x1, x2), 1)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    from datasets import load_datasets
    from config import cfg
    model = FasterGTA()
    train_loader, test_loader = load_datasets(cfg)
    data = next(iter(train_loader))
    img1 = data["previmg"]
    img2 = data["currimg"]
    out = model(img1, img2)
    print(out.shape)
    