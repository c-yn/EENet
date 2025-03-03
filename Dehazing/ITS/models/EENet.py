
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import *


class EBlock1(nn.Module):
    def __init__(self, out_channel, num_res=8):
        super(EBlock1, self).__init__()

        self.layers = UNet(out_channel, out_channel, num_res)
    def forward(self, x):
        return self.layers(x)


class DBlock1(nn.Module):
    def __init__(self, channel, num_res=8):
        super(DBlock1, self).__init__()

        self.layers = UNet(channel, channel, num_res)
    def forward(self, x):
        return self.layers(x)

class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(3, out_plane//4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane, kernel_size=1, stride=1, relu=False),
            nn.InstanceNorm2d(out_plane, affine=True)
        )

    def forward(self, x):
        x = self.main(x)
        return x


def FFT(x):
    x = torch.fft.fft2(x, dim=(-2,-1))
    amp = torch.abs(x)
    pha = torch.angle(x)
    return amp, pha

def IFFT(amp, pha):
    real = amp * torch.cos(pha)
    imag = amp * torch.sin(pha)
    return torch.fft.ifft2(torch.complex(real, imag), dim=(-2,-1))

class Fre_AP(nn.Module):
    def __init__(self, channel, num_res=1) -> None:
        super().__init__()


        self.conv_amp = BasicConv(channel, channel, kernel_size=1, stride=1)
        self.conv_pha = BasicConv(channel, channel, kernel_size=1, stride=1)

    def forward(self, x):

        x1_amp, x1_pha = FFT(x)
        x1_amp = self.conv_amp(x1_amp)
        x1_pha = self.conv_pha(x1_pha)
        x = torch.abs(IFFT(x1_amp, x1_pha))

        return x
    
class Fre_RI(nn.Module):
    def __init__(self, channel) -> None:
        super().__init__()
        self.conv_real = BasicConv(channel, channel, kernel_size=1, stride=1)
        self.conv_imag = BasicConv(channel, channel, kernel_size=1, stride=1)

    def forward(self, x):
        x = torch.fft.fft2(x, norm='backward')
        x_real = x.real
        x_imag = x.imag
        x_real = self.conv_real(x_real)
        x_imag = self.conv_imag(x_imag)
        x = torch.complex(x_real, x_imag)
        x = torch.fft.ifft2(x, dim=(-2,-1), norm='backward')
        return torch.abs(x)

class ConvFreBranch(nn.Module):
    def __init__(self, channel) -> None:
        super().__init__()

        self.fre_ap = Fre_AP(channel)
        self.fre_ri = Fre_RI(channel)


    def forward(self, x):
        out = self.fre_ri(x)
        out = self.fre_ap(out)
        return out

class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel*2, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        return self.merge(torch.cat([x1, x2], dim=1))

class SpaFre(nn.Module):
    def __init__(self, channel) -> None:
        super().__init__()

        self.spatial_scale = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
        )
        self.fre_scale = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, spa, fre):
        fre = self.spatial_scale(spa) + fre
        spa = self.fre_scale(fre) + spa
        return spa, fre 

class MIMOUNet(nn.Module):
    def __init__(self, num_res=8):
        super(MIMOUNet, self).__init__()

        base_channel = 32

        self.inter = nn.ModuleList([
            SpaFre(base_channel),
            SpaFre(base_channel*2),
            SpaFre(base_channel*4),
            SpaFre(base_channel*4),
            SpaFre(base_channel*2),
            SpaFre(base_channel)            
        ])

        self.FreEncoder = nn.ModuleList([
            ConvFreBranch(base_channel),
            ConvFreBranch(base_channel*2),
            ConvFreBranch(base_channel*4)            
        ])
        
        self.FreDecoder = nn.ModuleList([
            ConvFreBranch(base_channel*4),
            ConvFreBranch(base_channel*2),
            ConvFreBranch(base_channel)            
        ])

        self.Encoder = nn.ModuleList([
            EBlock1(base_channel, num_res),
            EBlock1(base_channel*2, num_res),
            EBlock1(base_channel*4, num_res),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.fre_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock1(base_channel * 4, num_res),
            DBlock1(base_channel * 2, num_res),
            DBlock1(base_channel, num_res)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.FreConvs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)

    def forward(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        outputs = list()
        # 256-spa
        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)
        # 256-fre
        x_fre = self.fre_extract[0](x)
        fre_res1 = self.FreEncoder[0](x_fre)

        res1, fre_res1 = self.inter[0](res1, fre_res1)

        # 128-spa
        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)
        # 128-fre
        z_fre = self.fre_extract[1](fre_res1)
        fre_res2 = self.FreEncoder[1](z_fre)

        res2, fre_res2 = self.inter[1](res2, fre_res2)

        # 64-spa
        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)
        # 64-fre
        z_fre = self.fre_extract[2](fre_res2)
        z_fre = self.FreEncoder[2](z_fre)
        # bottleneck
        z, z_fre = self.inter[2](z, z_fre)

        z = self.Decoder[0](z)

        z_fre = self.FreDecoder[0](z_fre)

        z_ = self.ConvsOut[0](z)
        z, z_fre = self.inter[3](z, z_fre)

        # 128-spa
        z = self.feat_extract[3](z)
        outputs.append(z_+x_4)

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        # 128-fre

        z_fre = self.fre_extract[3](z_fre)
        z_fre = torch.cat((z_fre, fre_res2), dim=1)
        z_fre = self.FreConvs[0](z_fre)

        z = self.Decoder[1](z)

        z_fre = self.FreDecoder[1](z_fre)
        z_ = self.ConvsOut[1](z)
        z, z_fre = self.inter[4](z, z_fre)

        # 256
        z = self.feat_extract[4](z)
        outputs.append(z_+x_2)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)

        z_fre = self.fre_extract[4](z_fre)
        z_fre = torch.cat((z_fre, fre_res1), dim=1)
        z_fre = self.FreConvs[1](z_fre)

        z = self.Decoder[2](z)

        z_fre = self.FreDecoder[2](z_fre)

        z, z_fre = self.inter[5](z, z_fre)

        z = self.feat_extract[5](z)
        z_fre = self.fre_extract[5](z_fre)

        outputs.append(z+x+z_fre)
        return outputs



def build_net():

    return MIMOUNet()
