import torch.nn as nn

from ops import *

class Unet(nn.Module):
    def __init__(self, in_channels, n_base_filters=32, layers=4, n_class=2):
        """
        The Unet architecture
        """
        super().__init__()
        self.maxpool = nn.MaxPool2d(2, 2)

        down_blocks = []
        up_blocks = []
        for i in range(layers):
            in_filters = n_base_filters * 2 ** (i-1) if i > 0 else in_channels
            out_filters = n_base_filters * 2 ** i

            down_blocks.append(ContextBlock(in_filters, out_filters))

            if i > 0:
                up_blocks.append(UpBlock(out_filters, in_filters))
        
        self.down_blocks = nn.ModuleList(down_blocks)
        self.up_blocks = nn.ModuleList(up_blocks[::-1])
        self.conv_seg = nn.Conv2d(in_channels = n_base_filters, out_channels=n_class, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        skip_connections = []
        for idx, down_block in enumerate(self.down_blocks):
            x = down_block(x)

            if idx < len(self.down_blocks) - 1:
                skip_connections.append(x)
                x = self.maxpool(x)

        for idx, up_block in enumerate(self.up_blocks):
            x = up_block(x, skip_connections[::-1][idx])

        x = self.conv_seg(x)

        return x
