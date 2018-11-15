import torch.nn as nn
import torch
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F
from torch.nn.parameter import Parameter

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'BasicBlock', 'Bottleneck', 'ResNet_new', 'BasicBlockSep', 'BasicBlockGroup', 'ResNet_combine']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class BasicBlockSep(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlockSep, self).__init__()
        #self.conv1 = conv3x3(inplanes, planes, stride)
        self.conv1_1 = nn.Conv2d(inplanes, int((inplanes+planes)/2), kernel_size=(3, 1), stride=(stride, 1),
                     padding=(1, 0), bias=False)
        self.conv1_2 = nn.Conv2d(int((inplanes+planes)/2), planes, kernel_size=(1, 3), stride=(1, stride),
                     padding=(0, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        #self.conv2 = conv3x3(planes, planes)
        self.conv2_1 = nn.Conv2d(planes, planes, kernel_size=(3, 1), stride=(1, 1),
                     padding=(1, 0), bias=False)
        self.conv2_2 = nn.Conv2d(planes, planes, kernel_size=(1, 3), stride=(1, 1),
                     padding=(0, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1_1(x)
        out = self.conv1_2(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2_1(out)
        out = self.conv2_2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class BasicBlockGroup(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlockGroup, self).__init__()
        #self.conv1 = conv3x3(inplanes, planes, stride)
        self.conv1_1 = nn.Conv2d(inplanes, inplanes, kernel_size=1, stride=1,
                     padding=0, bias=False)
        self.conv1_2 = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=stride,
                     padding=1, bias=False, groups=inplanes)
        self.conv1_3 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1,
                     padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        #self.conv2 = conv3x3(planes, planes)
        self.conv2_1 = nn.Conv2d(planes, planes, kernel_size=1, stride=1,
                     padding=0, bias=False)
        self.conv2_2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                     padding=1, bias=False, groups=planes)
        self.conv2_3 = nn.Conv2d(planes, planes, kernel_size=1, stride=1,
                     padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1_1(x)
        out = self.conv1_2(out)
        out = self.conv1_3(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2_1(out)
        out = self.conv2_2(out)
        out = self.conv2_3(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Mask(nn.Module):

    def __init__(self, channel, width, height):
        super(Mask, self).__init__()
        self.mode = 'layer_dist'
        self.data = Parameter(torch.ones(channel, width, height))

    def forward(self, x):
        #return self.data * x
        if (self.mode == 'mask'):
            return 0.5*(self.data.abs() - (self.data - 1).abs() + 1) * x
        elif (self.mode == 'layer_dist'):
            out = 4 + (self.data - torch.mean(self.data))/torch.std(self.data)
            out = (0.5*(out.abs() - (out - 8).abs() + 8) + 0.1) * 16
            return out * x


class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.01, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.noise = torch.tensor(0.).to(device)

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma if self.is_relative_detach else self.sigma
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
        return sampled_noise

class ResNet_new(nn.Module):

    def __init__(self, block_t, layers_t, block_s, layers_s, break_point_t, break_point_s, num_classes=1000):
        super(ResNet_new, self).__init__()
        self.mode = None
        self.break_point_s = break_point_s
        self.break_point_t = break_point_t
        self.epoch = 0
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block_t, 64, layers_t[0])
        self.inplanes = 64
        self.layer1_s = self._make_layer(block_s, 64, layers_s[0])
        self.mask1 = Mask(64, 56, 56)
        self.mask2 = Mask(64, 56, 56)
        self.layer2 = self._make_layer(block_t, 128, layers_t[1], stride=2)
        self.inplanes = 64
        self.layer2_s = self._make_layer(block_s, 128, layers_s[1], stride=2)
        self.mask3 = Mask(128, 28, 28)
        self.mask4 = Mask(128, 28, 28)
        self.layer3 = self._make_layer(block_t, 256, layers_t[2], stride=2)
        self.inplanes = 128
        self.layer3_s = self._make_layer(block_s, 256, layers_s[2], stride=2)
        self.mask5 = Mask(256, 14, 14)
        self.mask6 = Mask(256, 14, 14)
        self.layer4 = self._make_layer(block_t, 512, layers_t[3], stride=2)
        self.inplanes = 256
        self.layer4_s = self._make_layer(block_s, 512, layers_s[3], stride=2)
        self.mask7 = Mask(512, 7, 7)
        self.mask8 = Mask(512, 7, 7)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        assert(block_t.expansion == block_s.expansion)
        self.fc = nn.Linear(512 * block_t.expansion, num_classes)
        self.noise = GaussianNoise()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        if (self.mode == 'mask'):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1[0:self.break_point_t[0]](x)
            noise = self.noise(x)
            x = noise + x - self.mask1(noise)
            x = self.layer1[self.break_point_t[0]:](x)
            x = noise + x - self.mask2(noise)

            x = self.layer2[0:self.break_point_t[1]](x)
            x = noise + x - self.mask3(noise)
            x = self.layer2[self.break_point_t[1]:](x)
            x = noise + x - self.mask4(noise)

            x = self.layer3[0:self.break_point_t[2]](x)
            x = noise + x - self.mask5(noise)
            x = self.layer3[self.break_point_t[2]:](x)
            x = noise + x - self.mask6(noise)

            x = self.layer4[0:self.break_point_t[3]](x)
            x = noise + x - self.mask7(noise)
            x = self.layer4[self.break_point_t[3]:](x)
            x = noise + x - self.mask8(noise)

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

            return x

        elif (self.mode == 'layer_dist'):
            x0 = self.conv1(x)
            x0 = self.bn1(x0)
            x0 = self.relu(x0)
            x1 = self.maxpool(x0)
            
            x2 = self.layer1[0:self.break_point_t[0]](x1)
            x3 = self.layer1[self.break_point_t[0]:](x2)

            x4 = self.layer2[0:self.break_point_t[1]](x3)
            x5 = self.layer2[self.break_point_t[1]:](x4)

            x6 = self.layer3[0:self.break_point_t[2]](x5)
            x7 = self.layer3[self.break_point_t[2]:](x6)

            x8 = self.layer4[0:self.break_point_t[3]](x7)
            x9 = self.layer4[self.break_point_t[3]:](x8)

            x10 = self.avgpool(x9)
            x10 = x10.view(x10.size(0), -1)
            x11 = self.fc(x10) 

            x2_s = self.layer1_s[0:self.break_point_s[0]](x1)
            if (self.epoch > 2):
                x3_s = self.layer1_s[self.break_point_s[0]:](x2_s)
            else:
                x3_s = self.layer1_s[self.break_point_s[0]:](x2)
            if (self.epoch > 5):
                x4_s = self.layer2_s[0:self.break_point_s[1]](x3_s) 
            else:
                x4_s = self.layer2_s[0:self.break_point_s[1]](x3)
            if (self.epoch > 2):
                x5_s = self.layer2_s[self.break_point_s[1]:](x4_s)
            else:
                x5_s = self.layer2_s[self.break_point_s[1]:](x4)
            if (self.epoch > 8):
                x6_s = self.layer3_s[0:self.break_point_s[2]](x5_s)
            else:
                x6_s = self.layer3_s[0:self.break_point_s[2]](x5)
            if (self.epoch > 2):
                x7_s = self.layer3_s[self.break_point_s[2]:](x6_s)
            else:
                x7_s = self.layer3_s[self.break_point_s[2]:](x6)
            if (self.epoch > 5): 
                x8_s = self.layer4_s[0:self.break_point_s[3]](x7_s)
            else:               
                x8_s = self.layer4_s[0:self.break_point_s[3]](x7)
            if (self.epoch > 2):
                x9_s = self.layer4_s[self.break_point_s[3]:](x8_s)
            else:
                x9_s = self.layer4_s[self.break_point_s[3]:](x8)

            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1_s(x)
            x = self.layer2_s(x)
            x = self.layer3_s(x)
            x = self.layer4_s(x)

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

            return x, [x2, x3, x4, x5, x6, x7, x8, x9, x2_s, x3_s, x4_s, x5_s, x6_s, x7_s, x8_s, x9_s]

        elif (self.mode == 'test'):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1_s(x)
            x = self.layer2_s(x)
            x = self.layer3_s(x)
            x = self.layer4_s(x)

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            
            return x

class ResNet_combine(nn.Module):

    def __init__(self, block_t, layers_t, block_s, layers_s, break_point_t, break_point_s, num_classes=1000):
        super(ResNet_combine, self).__init__()
        self.mode = None
        self.break_point_s = break_point_s
        self.break_point_t = break_point_t
        self.epoch = 0
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv1_s = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1_s = nn.BatchNorm2d(64)
        self.mask1 = Mask(64, 112, 112)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block_t, 64, layers_t[0])
        self.inplanes = 64
        self.layer1_s = self._make_layer(block_s, 64, layers_s[0])
        self.mask2 = Mask(64, 56, 56)
        self.mask3 = Mask(64, 56, 56)
        self.layer2 = self._make_layer(block_t, 128, layers_t[1], stride=2)
        self.inplanes = 64
        self.layer2_s = self._make_layer(block_s, 128, layers_s[1], stride=2)
        self.mask4 = Mask(128, 28, 28)
        self.mask5 = Mask(128, 28, 28)
        self.layer3 = self._make_layer(block_t, 256, layers_t[2], stride=2)
        self.inplanes = 128
        self.layer3_s = self._make_layer(block_s, 256, layers_s[2], stride=2)
        self.mask6 = Mask(256, 14, 14)
        self.mask7 = Mask(256, 14, 14)
        self.layer4 = self._make_layer(block_t, 512, layers_t[3], stride=2)
        self.inplanes = 256
        self.layer4_s = self._make_layer(block_s, 512, layers_s[3], stride=2)
        self.mask8 = Mask(512, 7, 7)
        self.mask9 = Mask(512, 7, 7)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        assert(block_t.expansion == block_s.expansion)
        self.fc = nn.Linear(512 * block_t.expansion, num_classes)

        #self.fc_s = nn.Linear(512 * block_t.expansion, num_classes)

        self.l1_loss = nn.L1Loss()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        if (self.mode == 'mask'):
            x1 = self.conv1(x)
            x1 = self.bn1(x1)
            x2 = self.conv1_s(x)
            x2 = self.bn1_s(x2)
            x = x1 + x2 - self.mask1(x2)

            x = self.relu(x)
            x = self.maxpool(x)
            x1 = self.layer1[0:self.break_point_t[0]](x)
            x2 = self.layer1_s[0:self.break_point_s[0]](x)
            x = x1 + x2 - self.mask2(x2)
            x1 = self.layer1[self.break_point_t[0]:](x)
            x2 = self.layer1_s[self.break_point_s[0]:](x)
            x = x1 + x2 - self.mask3(x2)

            x1 = self.layer2[0:self.break_point_t[1]](x)
            x2 = self.layer2_s[0:self.break_point_s[1]](x)
            x = x1 + x2 - self.mask4(x2)
            x1 = self.layer2[self.break_point_t[1]:](x)
            x2 = self.layer2_s[self.break_point_s[1]:](x)
            x = x1 + x2 - self.mask5(x2)

            x1 = self.layer3[0:self.break_point_t[2]](x)
            x2 = self.layer3_s[0:self.break_point_s[2]](x)
            x = x1 + x2 - self.mask6(x2)
            x1 = self.layer3[self.break_point_t[2]:](x)
            x2 = self.layer3_s[self.break_point_s[2]:](x)
            x = x1 + x2 - self.mask7(x2)

            x1 = self.layer4[0:self.break_point_t[3]](x)
            x2 = self.layer4_s[0:self.break_point_s[3]](x)
            x = x1 + x2 - self.mask8(x2)
            x1 = self.layer4[self.break_point_t[3]:](x)
            x2 = self.layer4_s[self.break_point_s[3]:](x)
            x = x1 + x2 - self.mask9(x2)

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

            return x

        elif (self.mode == 'layer_dist'):
            x0 = self.conv1(x)
            x0 = self.bn1(x0)

            x0_s = self.conv1_s(x)
            x0_s = self.bn1_s(x0_s)

            x1 = self.relu(x0)
            x1 = self.maxpool(x1)

            x1_s = self.relu(x0_s)
            x1_s = self.maxpool(x1_s)
            
            x2 = self.layer1[0:self.break_point_t[0]](x1)
            x3 = self.layer1[self.break_point_t[0]:](x2)

            x4 = self.layer2[0:self.break_point_t[1]](x3)
            x5 = self.layer2[self.break_point_t[1]:](x4)

            x6 = self.layer3[0:self.break_point_t[2]](x5)
            x7 = self.layer3[self.break_point_t[2]:](x6)

            x8 = self.layer4[0:self.break_point_t[3]](x7)
            x9 = self.layer4[self.break_point_t[3]:](x8)

            x10 = self.avgpool(x9)
            x10 = x10.view(x10.size(0), -1)
            x10 = self.fc(x10) 

            if (self.epoch > 2):
                x2_s = self.layer1_s[0:self.break_point_s[0]](x1_s)
            else:
                x2_s = self.layer1_s[0:self.break_point_s[0]](x1)

            if (self.epoch > 2):
                x3_s = self.layer1_s[self.break_point_s[0]:](x2_s)
            else:
                x3_s = self.layer1_s[self.break_point_s[0]:](x2)
            if (self.epoch > 5):
                x4_s = self.layer2_s[0:self.break_point_s[1]](x3_s) 
            else:
                x4_s = self.layer2_s[0:self.break_point_s[1]](x3)
            if (self.epoch > 2):
                x5_s = self.layer2_s[self.break_point_s[1]:](x4_s)
            else:
                x5_s = self.layer2_s[self.break_point_s[1]:](x4)
            if (self.epoch > 8):
                x6_s = self.layer3_s[0:self.break_point_s[2]](x5_s)
            else:
                x6_s = self.layer3_s[0:self.break_point_s[2]](x5)
            if (self.epoch > 2):
                x7_s = self.layer3_s[self.break_point_s[2]:](x6_s)
            else:
                x7_s = self.layer3_s[self.break_point_s[2]:](x6)
            if (self.epoch > 5): 
                x8_s = self.layer4_s[0:self.break_point_s[3]](x7_s)
            else:               
                x8_s = self.layer4_s[0:self.break_point_s[3]](x7)
            if (self.epoch > 2):
                x9_s = self.layer4_s[self.break_point_s[3]:](x8_s)
            else:
                x9_s = self.layer4_s[self.break_point_s[3]:](x8)

            x = self.conv1_s(x)
            x = self.bn1_s(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1_s(x)
            x = self.layer2_s(x)
            x = self.layer3_s(x)
            x = self.layer4_s(x)

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

            return x, [x0, x0_s, x2, x2_s, x3, x3_s, x4, x4_s, x5, x5_s, x6, x6_s, x7, x7_s, x8, x8_s, x9, x9_s]

        elif (self.mode == 'test'):
            x = self.conv1_s(x)
            x = self.bn1_s(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1_s(x)
            x = self.layer2_s(x)
            x = self.layer3_s(x)
            x = self.layer4_s(x)

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            
            return x

class GroupConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size=7, stride=1, padding=1, bias=False):
        super(GroupConv, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, 12, kernel_size=kernel_size, stride=stride,
                     padding=padding, bias=False, groups=3)
        self.conv2 = nn.Conv2d(12, out_planes, kernel_size=1, stride=1,
                     padding=0, bias=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out

class SepConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size=7, stride=1, padding=1, bias=False):
        super(SepConv, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, 12, kernel_size=(kernel_size, 1), stride=(stride, 1),
                     padding=(padding, 0), bias=False)
        self.conv2 = nn.Conv2d(12, out_planes, kernel_size=(1, kernel_size), stride=(1, stride),
                     padding=(0, padding), bias=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model

