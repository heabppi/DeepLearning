import torch
import torch.nn as nn

# ResidualBlock (path1)만들기
class ResidualBlockUp(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(ResidualBlock, self).__init__()


    self.path1 = nn.Sequential(
          nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                    kernel_size=3, padding=1),
          nn.ReLU(),
          nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                    kernel_size=3, padding=1)
        )
    self.identity = nn.Identity()
    self.relu = nn.ReLU()

  def forward(self,x):
    identity = self.identity(x)

    x = self.path1(x)
    x += identity
    output = self.relu(x)

    return output

# ResidualBlockDown (path2) 만들기
class ResidualBlockDown(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(ResidualBlockDown, self).__init__()

    self.path1 = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                  kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                  kernel_size=3, padding=1)
    )
    self.path2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                  kernel_size=1, stride=2)
    self.relu = nn.ReLU()

  def forward(self,x):
    output2 = self.path2(x)

    x = self.path1(x)
    x += output2
    output = self.relu(x)

    return output

# ResidualBlock(path1 + path2)
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.path1 = nn.Sequential(
          nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                    kernel_size=3, stride=stride, padding=1),
          nn.ReLU(),
          nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                    kernel_size=3, stride=1, padding=1)
        )
        if stride == 1:
          self.path2 = nn.Identity()
        else:
          self.path2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                    kernel_size=1, stride=stride, padding=0)
        self.relu = nn.ReLU()

    def forward(self,x):
      output2 = self.path2(x)

      output1 = self.path1(x)
      output = output1 + output2
      output = self.relu(output)

      return output
    

# ResNet model 만들기
class ResNet(nn.Module):            #64           #128          #256          #512
  def __init__(self, in_channels, out_channel1, out_channel2, out_channel3, out_channel4):
    super(ResNet, self).__init__()

    self.layer1 = nn.Sequential(
        nn.Conv2d(in_channels, out_channel1, kernel_size=7, stride=2, padding=3),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

    self.layer2 = nn.Sequential(
        ResidualBlock(out_channel1, out_channel1),
        ResidualBlock(out_channel1, out_channel1),
        ResidualBlock(out_channel1, out_channel1)        
    )

    self.layer3 = nn.Sequential(
        ResidualBlock(out_channel1, out_channel2, stride=2),
        ResidualBlock(out_channel2, out_channel2),
        ResidualBlock(out_channel2, out_channel2),
        ResidualBlock(out_channel2, out_channel2)
    )

    self.layer4 = nn.Sequential(
        ResidualBlock(out_channel2, out_channel3, stride=2),
        ResidualBlock(out_channel3, out_channel3),
        ResidualBlock(out_channel3, out_channel3),
        ResidualBlock(out_channel3, out_channel3),
        ResidualBlock(out_channel3, out_channel3),
        ResidualBlock(out_channel3, out_channel3)
    )

    self.layer5 = nn.Sequential(
        ResidualBlock(out_channel3, out_channel4, stride=2),
        ResidualBlock(out_channel4, out_channel4),
        ResidualBlock(out_channel4, out_channel4)
    )

    self.layer6 = nn.Sequential(
        nn.AvgPool2d(kernel_size=7, padding=0),
        nn.Flatten(),
        nn.Linear(in_features=512*1*1, out_features=1000),
    )

  def forward(self, x):
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.layer5(x)
    x = self.layer6(x)
    return x
  
# 구현
import torch
input_tensor = torch.randn(size=(1,3,224, 224))

model = ResNet(3, 64, 128, 256, 512)
output = model(input_tensor)
print(output.shape)

# ResNet 만들기
class ResNet34(nn.Module):
  def __init__(self, in_channels, out_channel1, out_channel2, out_channel3, out_channel4):
    super(ResNet34, self).__init__()

    self.layer1 = nn.Sequential(
        nn.Conv2d(in_channels, out_channel1, kernel_size=7, stride=2, padding=3),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

    self.layer2 = self._make_layers(out_channel1, out_channel1, n_blocks=3, downsample=False)
    self.layer3 = self._make_layers(out_channel1, out_channel2, n_blocks=4, downsample=True)
    self.layer4 = self._make_layers(out_channel2, out_channel3, n_blocks=6, downsample=True)
    self.layer5 = self._make_layers(out_channel3, out_channel4, n_blocks=3, downsample=True)
    self.layer6 = nn.Sequential(
        nn.AvgPool2d(kernel_size=7, padding=0),
        nn.Flatten(),
        nn.Linear(in_features=512*1*1, out_features=1000),
    )


  def _make_layers(self, in_channels, out_channels, n_blocks, downsample):
    layers = []
    for _ in range(n_blocks):
      if downsample:
        layers.append(ResidualBlock(in_channels, out_channels, stride = 2))
        downsample = False
      else:
        layers.append(ResidualBlock(in_channels, out_channels))
      return nn.Sequential(*layers)
  

  def forward(self, x):
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.layer5(x)
    x = self.layer6(x)
    return x
  

#### ResNet18
class ResNet(nn.Module):
  def __init__(self, n_block_list, in_channels=3, out_channel1=64, out_channel2=128, out_channel3=256, out_channel4=512):
    super(ResNet34, self).__init__()

    self.layer1 = nn.Sequential(
        nn.Conv2d(in_channels, out_channel1, kernel_size=7, stride=2, padding=3),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

    self.layer2 = self._make_layers(out_channel1, out_channel1, n_blocks=n_block_list[0], downsample=False)
    self.layer3 = self._make_layers(out_channel1, out_channel2, n_blocks=n_block_list[1], downsample=True)
    self.layer4 = self._make_layers(out_channel2, out_channel3, n_blocks=n_block_list[2], downsample=True)
    self.layer5 = self._make_layers(out_channel3, out_channel4, n_blocks=n_block_list[3], downsample=True)
    self.layer6 = nn.Sequential(
        nn.AvgPool2d(kernel_size=7, padding=0),
        nn.Flatten(),
        nn.Linear(in_features=512*1*1, out_features=1000),
    )


  def _make_layers(self, in_channels, out_channels, n_blocks, downsample):
    layers = []
    for _ in range(n_blocks):
      if downsample:
        layers.append(ResidualBlock(in_channels, out_channels, stride = 2))
        downsample = False
      else:
        layers.append(ResidualBlock(in_channels, out_channels))
      return nn.Sequential(*layers)
  

  def forward(self, x):
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.layer5(x)
    x = self.layer6(x)
    return x
  
def ResNet18():
  return ResNet(n_block_list=[2,2,2,2])