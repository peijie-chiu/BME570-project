import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, activation='lrelu', normalization='instance'):
        super().__init__()
        self.normalization = normalization
        self.activation = activation

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)    
        if normalization == 'instance':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif normalization == 'batch':
            self.norm = nn.BatchNorm2d(out_channels)
        if activation == 'lrelu':
            self.act = nn.LeakyReLU(inplace=True)
        elif activation == 'relu':
            self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.normalization:
            x = self.norm(x)

        if self.activation:
            x = self.act(x)
        
        return x


class ContextBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, activation='lrelu', normalization='instance', drop_p=0.3):
        super().__init__()
        
        self.conv_block_1 = ConvBlock(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, activation=activation, normalization=normalization)
        self.dropout = nn.Dropout2d(p=drop_p)
        self.conv_block_2 = ConvBlock(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, activation=activation, normalization=normalization)

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.dropout(x)
        x = self.conv_block_2(x)
    
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation='lrelu', normalization='instance', interpolation="nearest"):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode=interpolation)
        self.conv_up = ConvBlock(in_channels, out_channels, activation=activation, normalization=normalization)
        self.local = ContextBlock(in_channels, out_channels)

    def forward(self, x, cat):
        x = self.upsample(x)
        x = self.conv_up(x)
        x = torch.cat([x, cat], dim=1)
        x = self.local(x)

        return x
