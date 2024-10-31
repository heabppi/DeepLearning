import torch
import torch.nn as nn

class ConvBlock(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
    super(ConvBlock,self).__init__()
    self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
    self.relu = nn.ReLU()

  def forward(self, x):
    x = self.conv(x)
    x = self.relu(x)
    return x
  

class InceptionNaive(nn.Module):
  def __init__(self, in_channels, out_channels1, out_channels2, out_channels3):
    super(InceptionNaive, self).__init__()

    self.conv1 = ConvBlock(in_channels, out_channels1, kernel_size=1, stride=1, padding=0)
    self.conv2 = ConvBlock(in_channels, out_channels2, kernel_size=3, stride=1, padding=1)
    self.conv3 = ConvBlock(in_channels, out_channels3, kernel_size=5, stride=1, padding=2)
    self.conv4 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

  def forward(self, x):
      x1 = self.conv1(x)
      x2 = self.conv2(x)
      x3 = self.conv3(x)
      x4 = self.conv4(x)
      
      output = torch.cat([x1, x2, x3, x4], dim=1)
      return output
  

class Inception(nn.Module):
  def __init__(self, in_channels, out_channels1, out_channels2, out_channels3, out_channels4, out_channels5, out_channels6):
    super(Inception, self).__init__()

    self.conv1 = ConvBlock(in_channels, out_channels1, kernel_size=1, stride=1, padding=0)

    self.conv2 = nn.Sequential(
        ConvBlock(in_channels, out_channels2, kernel_size=1, stride=1, padding=0),
        ConvBlock(out_channels2, out_channels3, kernel_size=3, stride=1, padding=1)
        )

    self.conv3 = nn.Sequential(
        ConvBlock(in_channels, out_channels4, kernel_size=1, stride=1, padding=0),
        ConvBlock(out_channels4, out_channels5, kernel_size=5, stride=1, padding=2),
        )

    self.conv4 = nn.Sequential(
        nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        ConvBlock(in_channels, out_channels6, kernel_size=1, stride=1, padding=0),
        )

  def forward(self, x):
      x1 = self.conv1(x)
      x2 = self.conv2(x)
      x3 = self.conv3(x)
      x4 = self.conv4(x)
      
      output = torch.cat([x1, x2, x3, x4], dim=1)
      return output
  

class GoogLeNet(nn.Module):
  def __init__(self):
    super(GoogLeNet, self).__init__()

    self.layer1 = nn.Sequential(
        ConvBlock(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    ) 

    self.layer2 = nn.Sequential(
        ConvBlock(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
        ConvBlock(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1),
    )

    self.layer3 = nn.Sequential(
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        Inception(192, 64, 96, 128, 16, 32, 32),
        Inception(256, 128, 128, 192, 32, 96, 64)
    )

    self.layer4 = nn.Sequential(
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        Inception(480, 192, 96, 208, 16, 48, 64),
        Inception(512, 160, 112, 224, 24, 64, 64),
        Inception(512, 128, 128, 256, 24, 64, 64),
        Inception(512, 112, 144, 288, 32, 64, 64),
        Inception(528, 256, 160, 320, 32, 128, 128),
    )

    self.layer5 = nn.Sequential(
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        Inception(832, 256, 160, 320, 32, 128, 128),
        Inception(832, 384, 192, 384, 48, 128, 128),
    )

    self.layer6 = nn.Sequential(
        nn.AvgPool2d(kernel_size=7, stride=1, padding=0),
        nn.Dropout(p=0.4),
        nn.Flatten(),
        nn.Linear(in_features=1024, out_features=1000),
        nn.Softmax(dim=1)
    )

  def forward(self,x):
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.layer5(x)
    x = self.layer6(x)
    return x
  
  
import torch
input_tensor = torch.randn(size=(1,3,224,224))

model = GoogLeNet()
output = model(input_tensor)
print(output.shape)