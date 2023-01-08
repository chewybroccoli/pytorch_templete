import torch.nn as nn
from base import BaseModel

class VGG(nn.Module):
  def __init__(self, num_classes):
    super(VGG, self).__init__()
    self.layer1 = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=16, kernel_size = 3, padding = 1), # batch_size x 32 x 96 x 96
        nn.BatchNorm2d(num_features=16),
        nn.ReLU(),
        nn.Conv2d(in_channels=16, out_channels=16, kernel_size = 3, padding = 1), # batch_size x 32 x 96 x 96
        nn.BatchNorm2d(num_features=16),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2), # batch_size x 16 x 50 x 50
        nn.Dropout(p=0.25)
    )
    self.layer2 = nn.Sequential(
        nn.Conv2d(in_channels=16, out_channels=32, kernel_size = 3, padding = 1), # batch_size x 32 x 45 x 45
        nn.BatchNorm2d(num_features=32),
        nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size = 3, padding = 1), # batch_size x 32 x 45 x 45
        nn.BatchNorm2d(num_features=32),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2), # batch_size x 32 x 25 x 25 
        nn.Dropout(p=0.25)
    )
    self.layer3 = nn.Sequential(
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size = 3, padding = 1), # batch_size x 48 x 20 x 20
        nn.BatchNorm2d(num_features=64),
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size = 3, padding = 1), # batch_size x 48 x 20 x 20
        nn.BatchNorm2d(num_features=64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2), #batch_size x 12 x 12
        nn.Dropout(p=0.25)
    )
    self.layer4 = nn.Sequential(
        nn.Conv2d(in_channels=64, out_channels=96, kernel_size = 3, padding = 1), # batch_size x 32 x 18 x 18
        nn.BatchNorm2d(num_features=96),
        nn.ReLU(),
        nn.Conv2d(in_channels=96, out_channels=96, kernel_size = 3, padding = 1), # batch_size x 32 x 18 x 18
        nn.BatchNorm2d(num_features=96),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2), #batch_size x 6 x 6
    )
    self.fc = nn.Sequential(
        nn.Linear(in_features = 96*6*6, out_features = 256),
        nn.BatchNorm1d(num_features = 256),
        nn.ReLU(),
        nn.Linear(in_features = 256, out_features = num_classes),
        nn.BatchNorm1d(num_features = num_classes)
    )

  def forward(self, x):
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = x.reshape(x.size(0), 96*6*6)
    x = self.fc(x)
    return x