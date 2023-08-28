import torch
import torch.nn as nn

class Refiner(nn.Module):
  """ Refiner designed to upsample from a low level probability map into a pixel probability map  """

  def __init__(self):

    super(Refiner, self).__init__()

    self.conv = nn.Conv2d(in_channels=769 , out_channels=1, kernel_size=(3, 3), padding='same')
    self.relu = nn.ReLU()
    self.bn = nn.BatchNorm2d(1)
    self.W = nn.Linear(224, 224)
    self.sigmoid = nn.Sigmoid()
    self.W.bias.data.fill_(0.)

  def forward(self, x, fv):
    x = x.unsqueeze(0).unsqueeze(0)
    """
    Args:
      x:  low level probability map (14x14)
      fv[1]:  image tokens encoded at layer 1
      fv[2]:  image tokens encoded at layer 2
      fv[3]:  image tokens encoded at layer 3
      fv[4]:  image tokens encoded at layer 4
    """

    f = fv[3][1:]
    fv4 = torch.zeros((1, 768, 14, 14))
    for c in range(768):
      for i in range(14):
        for j in range(14):
          fv4[0][c][i][j] = f[(i*14)+j][c]
    
    x = torch.cat([x, fv4], dim=1)
    x = self.conv(x)
    x = self.relu(x)
    x = self.bn(x)
    x = nn.functional.interpolate(input=x, mode='bilinear', scale_factor=2)


    f = fv[2][1:]
    fv3 = torch.zeros((1, 768, 14, 14))
    for c in range(768):
      for i in range(14):
        for j in range(14):
          fv3[0][c][i][j] = f[(i*14)+j][c]
    fv3 = nn.functional.interpolate(input=fv3, mode='bilinear', scale_factor=2)

    x = torch.cat([x, fv3], dim=1)
    x = self.conv(x)
    x = self.relu(x)
    x = self.bn(x)
    x = nn.functional.interpolate(input=x, mode='bilinear', scale_factor=2)


    f = fv[1][1:]
    fv2 = torch.zeros((1, 768, 14, 14))
    for c in range(768):
      for i in range(14):
        for j in range(14):
          fv2[0][c][i][j] = f[(i*14)+j][c]
    fv2 = nn.functional.interpolate(input=fv2, mode='bilinear', scale_factor=4)
    x = torch.cat([x, fv2], dim=1)
    x = self.conv(x)
    x = self.relu(x)
    x = self.bn(x)
    x = nn.functional.interpolate(input=x, mode='bilinear', scale_factor=2)


    f = fv[0][1:]
    fv1 = torch.zeros((1, 768, 14, 14))
    for c in range(768):
      for i in range(14):
        for j in range(14):
          fv1[0][c][i][j] = f[(i*14)+j][c]
    fv1 = nn.functional.interpolate(input=fv1, mode='bilinear', scale_factor=8)
    x = torch.cat([x, fv1], dim=1)
    x = self.conv(x)
    x = self.relu(x)
    x = self.bn(x)
    x = nn.functional.interpolate(input=x, mode='bilinear', scale_factor=2)

    x = self.W(x)
    x = self.sigmoid(x)

    return x
