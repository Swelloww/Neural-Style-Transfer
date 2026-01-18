import torch
import torch.nn as nn
from torchvision import models

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, upsample=False, normalize=True, relu=True):
        super(ConvBlock, self).__init__()
        self.upsample = upsample
        self.block = nn.Sequential()
        if upsample:
            self.block.add_module("upsample", nn.Upsample(scale_factor=2, mode='nearest'))
        
        # 使用 ReflectionPad 避免边界伪影
        self.block.add_module("pad", nn.ReflectionPad2d(kernel_size // 2))
        self.block.add_module("conv", nn.Conv2d(in_channels, out_channels, kernel_size, stride))
        
        if normalize:
            self.block.add_module("norm", nn.InstanceNorm2d(out_channels, affine=True))
        if relu:
            self.block.add_module("relu", nn.ReLU(True))

    def forward(self, x):
        return self.block(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, stride=1, normalize=True, relu=True),
            ConvBlock(channels, channels, kernel_size=3, stride=1, normalize=True, relu=False)
        )

    def forward(self, x):
        return self.block(x) + x

class TransformerNet(nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        # 下采样层
        self.net = nn.Sequential(
            ConvBlock(3, 32, kernel_size=9, stride=1),
            ConvBlock(32, 64, kernel_size=3, stride=2),
            ConvBlock(64, 128, kernel_size=3, stride=2),
            # 残差层 (学习风格的笔触逻辑)
            ResidualBlock(128), ResidualBlock(128), ResidualBlock(128),
            ResidualBlock(128), ResidualBlock(128),
            # 上采样层
            ConvBlock(128, 64, kernel_size=3, upsample=True),
            ConvBlock(64, 32, kernel_size=3, upsample=True),
            ConvBlock(32, 3, kernel_size=9, normalize=False, relu=False)
        )

    def forward(self, x):
        return self.net(x)

# VGG 特征提取器
class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        
        for x in range(4): self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9): self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 18): self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(18, 27): self.slice4.add_module(str(x), vgg_pretrained_features[x])
        
        if not requires_grad:
            for param in self.parameters(): param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        return h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3