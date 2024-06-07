import torch
from torch import nn
import math

import torch
import numpy as np
import torchvision
from torch import nn
import torch.nn.functional as F
from torch.nn import init
class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.is_last is False:
            # out = F.normalize(out)
            out = F.relu(out, inplace=True)
            # out = self.dropout(out)
        return out

class ConvLayertanh(torch.nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
            super(ConvLayertanh, self).__init__()
            reflection_padding = int(np.floor(kernel_size / 2))
            self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
            self.dropout = nn.Dropout2d(p=0.5)
            self.is_last = is_last

        def forward(self, x):
            out = self.reflection_pad(x)
            out = self.conv2d(out)
            if self.is_last is False:
                # out = F.normalize(out)
                out = torch.tanh(out)
                # out = self.dropout(out)
            return out

class AE_net(nn.Module):
    def __init__(self,input_nc=2, output_nc=2):
        super(AE_net,self).__init__()
        kernel_size =3
        stride =1
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2)
        #input_block
        self.conv1 =ConvLayer(in_channels =input_nc, out_channels =64,kernel_size=kernel_size,stride=stride)
        self.conv2 =ConvLayer(in_channels =64, out_channels =64,kernel_size=kernel_size,stride=stride)
        self.conv3 = ConvLayer(in_channels=64, out_channels=64, kernel_size=kernel_size, stride=stride)
        self.conv4 = ConvLayer(in_channels=64, out_channels=64*2, kernel_size=kernel_size, stride=stride)
        self.conv5 = ConvLayer(in_channels=64*2,out_channels=64*2, kernel_size=kernel_size, stride=stride)
        self.conv6 = ConvLayer(in_channels=64*2, out_channels=64*4, kernel_size=kernel_size, stride=stride)

        #middle block
        self.conv7 = ConvLayer(in_channels=64*4, out_channels=64*4, kernel_size=kernel_size, stride=stride)
        self.conv8 = ConvLayer(in_channels=64*4, out_channels=64*4, kernel_size=kernel_size, stride=stride)

        #output_block
        self.conv9 = ConvLayer(in_channels=64*8, out_channels=64*4, kernel_size=kernel_size, stride=stride)
        self.conv10 = ConvLayer(in_channels=64*6, out_channels=64*4, kernel_size=kernel_size, stride=stride)
        self.conv11 = ConvLayer(in_channels=64*6, out_channels=64*2, kernel_size=kernel_size, stride=stride)
        self.conv12 = ConvLayer(in_channels=64*3, out_channels=64 * 2, kernel_size=kernel_size, stride=stride)
        self.conv13 = ConvLayer(in_channels=64 * 3, out_channels=64 * 1, kernel_size=kernel_size, stride=stride)
        self.conv14 = ConvLayer(in_channels=64 * 2, out_channels=64 * 1, kernel_size=kernel_size, stride=stride)

        #output_head
        self.out =  ConvLayertanh(in_channels=64 * 1, out_channels=output_nc, kernel_size=kernel_size, stride=stride)

    def forward(self, x ,feat_need=False):

        fs = []
        fd = []
        #input_block
        f1 = self.conv1(x) #[B,C,H,W]
        fs.append(f1)
        f2 = self.conv2(f1) #[B,C,H,W]
        fs.append(f2)
        f3 = self.pool(self.conv3(f2)) #[B,C,H/2,W/2]
        fs.append(f3)
        f4 = self.conv4(f3) #[B,2C,H/2,W/2]
        fs.append(f4)
        f5 = self.pool(self.conv5(f4))#[B,2C,H/4,W/4]
        fs.append(f5)
        f6 = self.conv6(f5) #[B,4C,H/4,W/4]
        fs.append(f6)

        # middle block
        f7 = self.conv7(f6) #[B,4C,H/4,W/4]
        f8 = self.conv8(f7) #[B,4C,H/4,W/4]



        # output_block
        f9 = self.conv9(torch.cat([f8, f6], dim=1)) #[B,4C,H/4,W/4]
        fd.append(f9)
        f10 = self.up(self.conv10(torch.cat([f9, f5], dim=1)))  #[B,4C,H/2,W/2]
        fd.append(f10)
        f11 = self.conv11(torch.cat([f10, f4], dim=1))  #[B,2C,H/2,W/2]
        fd.append(f11)
        f12 = self.up(self.conv12(torch.cat([f11, f3], dim=1)))  # [B,2C,H,W]
        fd.append(f12)
        f13 = self.conv13(torch.cat([f12, f2], dim=1))  # [B,2C,H/2,W/2]
        fd.append(f13)
        f14 = self.conv14(torch.cat([f13, f1], dim=1))  # [B,2C,H/2,W/2]
        fd.append(f14)

        output = self.out(f14)
        if feat_need:
            return fs, fd
        else :
            return output

input = torch.randn(1,2,160,160)
model =AE_net(input_nc=2,output_nc=2)
output = model(input,feat_need=True)
for i in output[1]:
    print(i.shape)