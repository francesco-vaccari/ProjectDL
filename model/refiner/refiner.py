import torch
import torch.nn as nn

class Refiner(nn.Module):
  """ Refiner designed to upsample from a low level probability map into a pixel probability map  """

  def __init__(self):

    super(Refiner, self).__init__()
    
    self.conv1 = nn.Conv2d(in_channels=769 , out_channels=1, kernel_size=(3, 3), padding='same')
    self.relu1 = nn.ReLU()
    self.bn1 = nn.BatchNorm2d(1)

    self.conv2 = nn.Conv2d(in_channels=769 , out_channels=1, kernel_size=(3, 3), padding='same')
    self.relu2 = nn.ReLU()
    self.bn2 = nn.BatchNorm2d(1)

    self.conv3 = nn.Conv2d(in_channels=769 , out_channels=1, kernel_size=(3, 3), padding='same')
    self.relu3 = nn.ReLU()
    self.bn3 = nn.BatchNorm2d(1)

    self.conv4 = nn.Conv2d(in_channels=769 , out_channels=1, kernel_size=(3, 3), padding='same')
    self.relu4 = nn.ReLU()
    self.bn4 = nn.BatchNorm2d(1)

    self.W = nn.Linear(224*224, 224*224)
    self.sigmoid = nn.Sigmoid()
    self.W.bias.data.fill_(0.)

  def forward(self, x, fv):
    
    """
    Args:
      x:  low level probability map (batch x 14 x 14)
      fv[0]:  image tokens encoded at layer 1   (batch x 196 x 768)
      fv[1]:  image tokens encoded at layer 2   (batch x 196 x 768)
      fv[2]:  image tokens encoded at layer 3   (batch x 196 x 768)
      fv[3]:  image tokens encoded at layer 4   (batch x 196 x 768)
    """

    bacth_size = fv[3].shape[0]
    n_channels = fv[3].shape[2]
    x = x.unsqueeze(1)


    fv4 = fv[3][:, 1:, :]
    fv4 = torch.transpose(fv4, 1, 2)
    fv4 = fv4.reshape(bacth_size, n_channels, 14, 14)
    x = torch.cat([x, fv4], dim=1)
    x = self.conv1(x)
    x = self.relu1(x)
    x = self.bn1(x)
    x = nn.functional.interpolate(input=x, mode='bilinear', scale_factor=2)


    fv3 = fv[2][:, 1:, :]
    fv3 = torch.transpose(fv3, 1, 2)
    fv3 = fv3.reshape(bacth_size, n_channels, 14, 14)
    fv3 = nn.functional.interpolate(input=fv3, mode='bilinear', scale_factor=2)
    x = torch.cat([x, fv3], dim=1)
    x = self.conv2(x)
    x = self.relu2(x)
    x = self.bn2(x)
    x = nn.functional.interpolate(input=x, mode='bilinear', scale_factor=2)


    fv2 = fv[1][:, 1:, :]
    fv2 = torch.transpose(fv2, 1, 2)
    fv2 = fv2.reshape(bacth_size, n_channels, 14, 14)
    fv2 = nn.functional.interpolate(input=fv2, mode='bilinear', scale_factor=4)
    x = torch.cat([x, fv2], dim=1)
    x = self.conv3(x)
    x = self.relu3(x)
    x = self.bn3(x)
    x = nn.functional.interpolate(input=x, mode='bilinear', scale_factor=2)


    fv1 = fv[0][:, 1:, :]
    fv1 = torch.transpose(fv1, 1, 2)
    fv1 = fv1.reshape(bacth_size, n_channels, 14, 14)
    fv1 = nn.functional.interpolate(input=fv1, mode='bilinear', scale_factor=8)
    x = torch.cat([x, fv1], dim=1)
    x = self.conv4(x)
    x = self.relu4(x)
    x = self.bn4(x)
    x = nn.functional.interpolate(input=x, mode='bilinear', scale_factor=2)

    print(x.shape)
    x = torch.flatten(x, start_dim=2)
    x = self.W(x)
    x = self.sigmoid(x)
    print(x.shape)
    x = x.reshape(bacth_size, 1, 224, 224)
    x = x.squeeze(1)

    return x
