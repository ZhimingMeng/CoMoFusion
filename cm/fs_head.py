import math

import torch
from torch import nn
import torch.nn.functional as F

from cm.se import ChannelSpatialSELayer

class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels*(1+self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1)
        return x


class Block(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.block = nn.Sequential(
                nn.Conv2d(dim, dim, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(dim, dim_out, 3, padding=1),
                nn.ReLU()
            )
    def forward(self, x):
        return self.block(x)



class AttentionBlock(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, padding=1),
            nn.ReLU(),
            ChannelSpatialSELayer(num_channels=dim_out, reduction_ratio=2)
        )

    def forward(self, x):
        return self.block(x)
class Headtanh2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(Headtanh2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)

    def forward(self, x):
        return torch.tanh(self.conv(x))   # (-1, 1)



class HeadLeakyRelu2d(nn.Module):
    # convolution
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(HeadLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)

    def forward(self, x):
        return F.leaky_relu(self.conv(x), negative_slope=0.2)



class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2       #64
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding

class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels*(1+self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1)
        return x


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class Fusion_Head(nn.Module):
    def __init__(self):
        super(Fusion_Head, self).__init__()

        self.conv1 = Block(64*4,64*4)  #CNNBlock
        self.att1 = AttentionBlock(64*4,64 * 2) #SCABlock

        self.conv2 = Block(64 * 2, 64 * 2)
        self.att2 = AttentionBlock(64 * 2, 64 * 1)


        self.conv3 = Block(64 * 1, 64 * 1)


        self.decode = Headtanh2d(64, 1)


    def forward(self, feats):


        for i in range(len(feats)):
          feats[i] = feats[i].type(torch.float32)

        f1 = feats[5]
        # c1 =f1
        print(f1.dtype)
        c1 = self.conv1(f1)
        a1 = self.att1(c1)
        u1 = F.interpolate(a1, scale_factor=2, mode="bilinear", align_corners=True) #[1,384,80,80]


        f2 =feats[3]
        c2 = self.conv2(f2)
        # c2 =f2
        a2 = self.att2(c2+u1)
        u2 = F.interpolate(a2, scale_factor=2, mode="bilinear", align_corners=True)  # [1,384,80,80]


        f3 =feats[0]
        c3 = self.conv3(f3)
        # c3 = f3
        x = c3 + u2


        # Fusion Head
        mask = self.decode(x)

        return mask




class Fusion_Head_backfs(nn.Module):
    def __init__(self):
        super(Fusion_Head_backfs, self).__init__()

        self.conv1 = Block(64*4,64*4)
        self.att1 = AttentionBlock(64*4,64 * 2)

        self.conv2 = Block(64 * 2, 64 * 2)
        self.att2 = AttentionBlock(64 * 2, 64 * 1)


        self.conv3 = Block(64 * 1, 64 * 1)


        self.decode = Headtanh2d(64, 1)


    def forward(self, feats):
        # h = x.type(self.dtype)
        for i in range(len(feats)):
          feats[i] = feats[i].type(torch.float32)


        f1 = feats[0]
        c1 = self.conv1(f1)
        a1 = self.att1(c1)
        u1 = F.interpolate(a1, scale_factor=2, mode="bilinear", align_corners=True) #[1,384,80,80]


        f2 =feats[2]
        c2 = self.conv2(f2)
        a2 = self.att2(c2+u1)
        u2 = F.interpolate(a2, scale_factor=2, mode="bilinear", align_corners=True)  # [1,384,80,80]


        f3 =feats[5]
        c3 = self.conv3(f3)
        x = c3 + u2


        # Fusion Head
        mask = self.decode(x)

        return mask


