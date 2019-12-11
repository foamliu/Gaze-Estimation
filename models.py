import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter
from torchscope import scope
from torchvision import models

from config import device, num_classes, emb_size


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class GDConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, padding, bias=False):
        super(GDConv, self).__init__()
        self.depthwise = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding, groups=in_planes,
                                   bias=bias)
        self.bn = nn.BatchNorm2d(in_planes)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn(x)
        return x


class MobileNetMatchModel(nn.Module):
    def __init__(self):
        super(MobileNetMatchModel, self).__init__()
        mobilenet = models.mobilenet_v2(pretrained=True)
        # Remove linear layer
        modules = list(mobilenet.children())[:-1]
        self.features = nn.Sequential(*modules)
        # building last several layers
        self.dw_conv = GDConv(in_planes=1280, out_planes=1280, kernel_size=7, padding=0)
        self.fc = nn.Linear(1280, 512)
        self.bn = nn.BatchNorm1d(512)

    def forward(self, x):
        x = self.features(x)
        x = self.dw_conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn(x)
        return x


class ResNetMatchModel(nn.Module):
    def __init__(self):
        super(ResNetMatchModel, self).__init__()
        resnet = models.resnet50(pretrained=True)
        # Remove linear layer
        modules = list(resnet.children())[:-1]
        self.features = nn.Sequential(*modules)
        # building last several layers
        self.fc = nn.Linear(2048, 512)
        self.bn = nn.BatchNorm1d(512)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn(x)
        return x


class ArcMarginModel(nn.Module):
    def __init__(self, args):
        super(ArcMarginModel, self).__init__()

        self.weight = Parameter(torch.FloatTensor(num_classes, emb_size))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = args.easy_margin
        self.m = args.margin_m
        self.s = args.margin_s

        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, input, label):
        x = F.normalize(input)
        W = F.normalize(self.weight)
        cosine = F.linear(x, W)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device=device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


if __name__ == "__main__":
    model = ResNetMatchModel()
    scope(model, input_size=(3, 224, 224))
