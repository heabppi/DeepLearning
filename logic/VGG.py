import torch
import torch.nn as nn
from torchsummary import summary

input_tensor = torch.randn(size=(224,224,3))


# VGG11 모델구현
class VGG11(nn.Module):
  def __init__(self):
    super(VGG11, self).__init__()

    self.layer1 = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
        nn.Tanh(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    ) # 112

    self.layer2 = nn.Sequential(
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
        nn.Tanh(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    ) # 56

    self.layer3 = nn.Sequential(
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
        nn.Tanh(),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
        nn.Tanh(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    ) # 28

    self.layer4 = nn.Sequential(
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
        nn.Tanh(),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
        nn.Tanh(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    ) # 14

    self.layer5 = nn.Sequential(
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
        nn.Tanh(),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
        nn.Tanh(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    ) # 7

    self.layer6 = nn.Sequential(
        nn.Linear(in_features=512 * 7 * 7, out_features=4096),
        nn.Tanh(),
        nn.Linear(in_features=4096, out_features=4096),
        nn.Tanh(),
        nn.Linear(in_features=4096, out_features=1000),
    )
    

  def forward(self, x):
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.layer5(x)
    x = x.view(x.size(0), -1)
    x = self.layer6(x)

    return x

model = VGG11()
summary(model, input_size=(3, 224, 224))


# VGG13모델 구현
class VGG13(nn.Module):
  def __init__(self):
    super(VGG13, self).__init__()

    self.layer1 = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
        nn.Tanh(),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
        nn.Tanh(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    ) # 112

    self.layer2 = nn.Sequential(
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
        nn.Tanh(),
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
        nn.Tanh(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    ) # 56

    self.layer3 = nn.Sequential(
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
        nn.Tanh(),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
        nn.Tanh(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    ) # 28

    self.layer4 = nn.Sequential(
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
        nn.Tanh(),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
        nn.Tanh(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    ) # 14

    self.layer5 = nn.Sequential(
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
        nn.Tanh(),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
        nn.Tanh(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    ) # 7

    self.layer6 = nn.Sequential(
        nn.Linear(in_features=512 * 7 * 7, out_features=4096),
        nn.Tanh(),
        nn.Linear(in_features=4096, out_features=4096),
        nn.Tanh(),
        nn.Linear(in_features=4096, out_features=1000),
    )
    

  def forward(self, x):
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.layer5(x)
    x = x.view(x.size(0), -1)
    x = self.layer6(x)

    return x
  

# VGG19모델
class VGG19(nn.Module):
  def __init__(self):
    super(VGG13, self).__init__()

    self.layer1 = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
        nn.Tanh(),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
        nn.Tanh(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    ) # 112

    self.layer2 = nn.Sequential(
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
        nn.Tanh(),
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
        nn.Tanh(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    ) # 56

    self.layer3 = nn.Sequential(
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
        nn.Tanh(),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
        nn.Tanh(),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
        nn.Tanh(),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
        nn.Tanh(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    ) # 28

    self.layer4 = nn.Sequential(
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
        nn.Tanh(),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
        nn.Tanh(),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
        nn.Tanh(),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
        nn.Tanh(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    ) # 14

    self.layer5 = nn.Sequential(
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
        nn.Tanh(),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
        nn.Tanh(),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
        nn.Tanh(),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
        nn.Tanh(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    ) # 7

    self.layer6 = nn.Sequential(
        nn.Linear(in_features=512 * 7 * 7, out_features=4096),
        nn.Tanh(),
        nn.Linear(in_features=4096, out_features=4096),
        nn.Tanh(),
        nn.Linear(in_features=4096, out_features=1000),
    )
    

  def forward(self, x):
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.layer5(x)
    x = x.view(x.size(0), -1)
    x = self.layer6(x)

    return x
  

# ConvBlock 만들기
import torch.nn as nn

class ConvBlock(nn.Module):
  def __init__(self, in_channels, out_channels, n_layers):
    super(ConvBlock, self).__init__()

    self.layers = []
    for _ in range(n_layers):
      self.layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1))
      self.layers.append(nn.ReLU())
      in_channels = out_channels

    self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

    self.layers = nn.Sequential(*self.layers)

  def forward(self, x):
    x = self.layers(x)
    return x
  
# VGG11_ConvBlock 사용
class VGG11(nn.Module):
  def __init__(self):
    super(VGG11, self).__init__()

    self.layer1 = ConvBlock(in_channels=3, out_channels=64, n_layers=1)
    self.layer2 = ConvBlock(in_channels=64, out_channels=128, n_layers=1)
    self.layer3 = ConvBlock(in_channels=128, out_channels=256, n_layers=2)
    self.layer4 = ConvBlock(in_channels=256, out_channels=512, n_layers=2)
    self.layer5 = ConvBlock(in_channels=512, out_channels=512, n_layers=2)

    self.layer6 = nn.Sequential(
        nn.Linear(in_features=512 * 7 * 7, out_features=4096),
        nn.Tanh(),
        nn.Linear(in_features=4096, out_features=4096),
        nn.Tanh(),
        nn.Linear(in_features=4096, out_features=1000),
    )
    

  def forward(self, x):
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.layer5(x)
    x = x.view(x.size(0), -1)
    x = self.layer6(x)

    return x
  
# VGG13_ConvBlock 사용
class VGG13(nn.Module):
  def __init__(self):
    super(VGG13, self).__init__()

    self.layer1 = ConvBlock(in_channels=3, out_channels=64, n_layers=2)
    self.layer2 = ConvBlock(in_channels=64, out_channels=128, n_layers=2)
    self.layer3 = ConvBlock(in_channels=128, out_channels=256, n_layers=2)
    self.layer4 = ConvBlock(in_channels=256, out_channels=512, n_layers=2)
    self.layer5 = ConvBlock(in_channels=512, out_channels=512, n_layers=2)

    self.layer6 = nn.Sequential(
        nn.Linear(in_features=512 * 7 * 7, out_features=4096),
        nn.Tanh(),
        nn.Linear(in_features=4096, out_features=4096),
        nn.Tanh(),
        nn.Linear(in_features=4096, out_features=1000),
    )
    

  def forward(self, x):
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.layer5(x)
    x = x.view(x.size(0), -1)
    x = self.layer6(x)

    return x
  
# VGG19_ConvBlock 사용
class VGG19(nn.Module):
  def __init__(self):
    super(VGG19, self).__init__()

    self.layer1 = ConvBlock(in_channels=3, out_channels=64, n_layers=2)
    self.layer2 = ConvBlock(in_channels=64, out_channels=128, n_layers=2)
    self.layer3 = ConvBlock(in_channels=128, out_channels=256, n_layers=4)
    self.layer4 = ConvBlock(in_channels=256, out_channels=512, n_layers=4)
    self.layer5 = ConvBlock(in_channels=512, out_channels=512, n_layers=4)

    self.layer6 = nn.Sequential(
        nn.Linear(in_features=512 * 7 * 7, out_features=4096),
        nn.Tanh(),
        nn.Linear(in_features=4096, out_features=4096),
        nn.Tanh(),
        nn.Linear(in_features=4096, out_features=1000),
    )
    

  def forward(self, x):
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.layer5(x)
    x = x.view(x.size(0), -1)
    x = self.layer6(x)

    return x