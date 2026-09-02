import torch
import torch.nn as nn

# Channel Attention Block
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y



# Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(self, n_feat, reduction=16, residual_scaling=1.0):
        super().__init__()
        self.res_scale = residual_scaling
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1, bias=True),
        )
        self.ca = CALayer(n_feat, reduction)

    def forward(self, x):
        res = self.body(x)
        res = self.ca(res)
        res = res * self.res_scale
        return res + x



# Residual Group
class ResidualGroup(nn.Module):
    def __init__(self, n_feat, n_rcab, reduction, residual_scaling):
        super().__init__()
        modules = [RCAB(n_feat, reduction, residual_scaling) for _ in range(n_rcab)]
        modules.append(nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1, bias=True))
        self.body = nn.Sequential(*modules)

    def forward(self, x):
        res = self.body(x)
        return res + x



# Upsampler (PixelShuffle)
class Upsampler(nn.Module):
    def __init__(self, scale, n_feat):
        super().__init__()
        modules = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(scale).bit_length() - 1):
                modules += [
                    nn.Conv2d(n_feat, 4 * n_feat, kernel_size=3, padding=1),
                    nn.PixelShuffle(2),
                    nn.ReLU(inplace=True)
                ]
        elif scale == 3:
            modules += [
                nn.Conv2d(n_feat, 9 * n_feat, kernel_size=3, padding=1),
                nn.PixelShuffle(3),
                nn.ReLU(inplace=True)
            ]
        else:
            raise ValueError(f"Unsupported scale: {scale}")
        self.body = nn.Sequential(*modules)

    def forward(self, x):
        return self.body(x)



# RCAN Network (With Upsampling)
class RCAN2D(nn.Module):
    def __init__(self, input_channels=3, num_features=64,
                 num_residual_groups=5, num_residual_blocks=10,
                 reduction=16, residual_scaling=1.0,
                 scale=2, output_channels=None):
        super().__init__()
        if output_channels is None:
            output_channels = input_channels

        # head
        self.head = nn.Conv2d(input_channels, num_features, kernel_size=3, padding=1)

        # body
        self.body = nn.Sequential(
            *[ResidualGroup(num_features, num_residual_blocks, reduction, residual_scaling)
              for _ in range(num_residual_groups)],
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        )

        # upsampler
        self.upsampler = Upsampler(scale, num_features)

        # tail
        self.tail = nn.Conv2d(num_features, output_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x_head = self.head(x)
        res = self.body(x_head)
        res = res + x_head
        out = self.upsampler(res)
        out = self.tail(out)
        return out