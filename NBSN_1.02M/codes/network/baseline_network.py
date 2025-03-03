import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class crop(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        N, C, H, W = x.shape
        x = x[0:N, 0:C, 0:H-1, 0:W] # crop last row
        return x

class shift(nn.Module):
    def __init__(self):
        super().__init__()
        self.shift_down = nn.ZeroPad2d((0,0,1,0))
        self.crop = crop()

    def forward(self, x):
        x = self.shift_down(x)
        x = self.crop(x)
        return x

class super_shift(nn.Module):
    def __init__(self): 
        super().__init__()

    def forward(self, x, hole_size=1):
        shift_offset = (hole_size+1)//2 # hole_size must be = 1, 3, 5, 7...

        x = nn.ZeroPad2d((0,0,shift_offset,0))(x) # left right top bottom
        N, C, H, W = x.shape
        x = x[0:N, 0:C, 0:H-shift_offset, 0:W] # crop last rows
        return x

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, blind=True,stride=1, padding=1, kernel_size=3):
        super().__init__()
        self.blind = blind
        if blind:
            self.shift_down = nn.ZeroPad2d((0,0,1,0)) # left right top bottom
            self.crop = crop()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias) 
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        if self.blind:
            x = self.shift_down(x)
        x = self.conv(x)
        if self.blind:
            x = self.crop(x)
        x = self.relu(x)        
        return x

class Pool(nn.Module):
    def __init__(self, blind=True):
        super().__init__()
        self.blind = blind
        if blind:
            self.shift = shift()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        if self.blind:
            x = self.shift(x)
        x = self.pool(x)
        return x

class rotate(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x90 = x.transpose(2,3).flip(3)
        x180 = x.flip(2).flip(3)
        x270 = x.transpose(2,3).flip(2)
        x = torch.cat((x,x90,x180,x270), dim=0)
        return x

class unrotate(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x0, x90, x180, x270 = torch.chunk(x, 4, dim=0)
        x90 = x90.transpose(2,3).flip(2)
        x180 = x180.flip(2).flip(3)
        x270 = x270.transpose(2,3).flip(3)
        x = torch.cat((x0,x90,x180,x270), dim=1)
        return x

class ENC_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, reduce=True, blind=True):
        super().__init__()
        self.reduce = reduce
        self.conv1 = Conv(in_channels, out_channels, bias=bias, blind=blind)
        if reduce:
            self.pool = Pool(blind=blind)

    def forward(self, x):
        x = self.conv1(x)
        if self.reduce:
            x = self.pool(x)
        return x

class DEC_Conv(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, bias=True, increase=True, blind=True):
        super().__init__()
        self.increase = increase
        self.conv1 = Conv(in_channels, mid_channels, bias=bias, blind=blind)
        self.conv2 = Conv(mid_channels, out_channels, bias=bias, blind=blind)
        if increase:
            self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x, x_in):
        x = torch.cat((x, x_in), dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        if self.increase:
            x = self.upsample(x)
        return x

class Blind_UNet(nn.Module):
    def __init__(self, n_channels=3, mid_channels=48, n_output=96, bias=True, blind=True):
        super().__init__()
        self.intro = Conv(n_channels, mid_channels, bias=bias, blind=blind)
        self.enc1 = ENC_Conv(mid_channels, mid_channels, bias=bias, blind=blind)
        self.enc2 = ENC_Conv(mid_channels, mid_channels, bias=bias, blind=blind)
        self.enc3 = ENC_Conv(mid_channels, mid_channels, bias=bias, blind=blind)
        self.enc4 = ENC_Conv(mid_channels, mid_channels, bias=bias, blind=blind)
        self.enc5 = ENC_Conv(mid_channels, mid_channels, bias=bias, blind=blind)
        self.enc6 = ENC_Conv(mid_channels, mid_channels, bias=bias, reduce=False, blind=blind)
        self.dec5 = DEC_Conv(mid_channels*2, mid_channels*2, mid_channels*2, bias=bias, blind=blind)
        self.dec4 = DEC_Conv(mid_channels*3, mid_channels*2, mid_channels*2, bias=bias, blind=blind)
        self.dec3 = DEC_Conv(mid_channels*3, mid_channels*2, mid_channels*2, bias=bias, blind=blind)
        self.dec2 = DEC_Conv(mid_channels*3, mid_channels*2, mid_channels*2, bias=bias, blind=blind)
        self.dec1 = DEC_Conv(mid_channels*2+n_channels, mid_channels*2, n_output, bias=bias, increase=False, blind=blind)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, input):
        x = self.intro(input)
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)
        x = self.enc6(x5)
        x = self.upsample(x)
        x = self.dec5(x, x4)
        x = self.dec4(x, x3)
        x = self.dec3(x, x2)
        x = self.dec2(x, x1)
        x = self.dec1(x, input)
        return x
        
class ATBSN(nn.Module):  # 1.27m
    def __init__(self, n_channels=3, mid_channels=48, n_output=3, bias=True, blind=True):
        super().__init__()
        self.blind = blind
        self.rotate = rotate()
        self.unet = Blind_UNet(n_channels=n_channels, mid_channels=mid_channels, n_output=mid_channels*2, bias=bias, blind=blind)
        self.shift = super_shift()
        self.unrotate = unrotate()
        self.nin_A = nn.Conv2d(mid_channels*8, mid_channels*8, 1, bias=bias)
        self.nin_B = nn.Conv2d(mid_channels*8, mid_channels*2, 1, bias=bias)
        self.nin_C = nn.Conv2d(mid_channels*2, n_output, 1, bias=bias)

        with torch.no_grad():
            self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, a=0.1)
                m.bias.data.zero_()
        nn.init.kaiming_normal_(self.nin_C.weight.data, nonlinearity="linear")
    
    def forward(self, x, hole_size=1):              
        x = self.rotate(x)
                
        x = self.unet(x)
        if self.blind:
            x = self.shift(x, hole_size)
        x = self.unrotate(x)

        x0 = F.leaky_relu_(self.nin_A(x), negative_slope=0.1)
        x0 = F.leaky_relu_(self.nin_B(x0), negative_slope=0.1)
        x0 = self.nin_C(x0)
            
        return x0
    
# class N_BSN(nn.Module): #student c, 1.00m (1.02m in the paper is a typo)
#     def __init__(self, n_channels=3, mid_channels=48, n_output=3, bias=True, blind=False):
#         super().__init__()
#         self.unet = Blind_UNet(n_channels=n_channels, mid_channels=mid_channels, n_output=n_output, bias=bias, blind=blind)
#
#         with torch.no_grad():
#             self._init_weights()

#     def _init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight.data, a=0.1)
#                 m.bias.data.zero_()
    
#     def forward(self, x):                              
#         x0 = self.unet(x)
            
#         return x0

class Conv2d_relu(nn.Module): 
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x 
    
class N_BSN(nn.Module): # 1.02m version as described in the AT-BSN paper.
    def __init__(self):
        super(N_BSN, self).__init__()
        self.enc1 = Conv2d_relu(in_channels=3, out_channels =48, kernel_size =3, stride=1, padding =1)
        self.enc2 = Conv2d_relu(in_channels=48, out_channels =48, kernel_size =3, stride=1, padding =1)
        self.enc3 = Conv2d_relu(in_channels=48, out_channels =48, kernel_size =3, stride=1, padding =1)
        self.enc4 = Conv2d_relu(in_channels=48, out_channels =48, kernel_size =3, stride=1, padding =1)
        self.enc5 = Conv2d_relu(in_channels=48, out_channels =48, kernel_size =3, stride=1, padding =1)
        self.enc6 = Conv2d_relu(in_channels=48, out_channels =48, kernel_size =3, stride=1, padding =1)
        
        self.dec5_2 = Conv2d_relu(in_channels=96, out_channels=96, kernel_size =3, stride=1, padding =1)
        self.dec5_1 = Conv2d_relu(in_channels=96, out_channels=96, kernel_size =3, stride=1, padding =1)
        self.dec4_2 = Conv2d_relu(in_channels=144, out_channels=96, kernel_size =3, stride=1, padding =1)
        self.dec4_1 = Conv2d_relu(in_channels=96, out_channels=96, kernel_size =3, stride=1, padding =1)
        self.dec3_2 = Conv2d_relu(in_channels=144, out_channels=96, kernel_size =3, stride=1, padding =1)
        self.dec3_1 = Conv2d_relu(in_channels=96, out_channels=96, kernel_size =3, stride=1, padding =1)
        self.dec2_2 = Conv2d_relu(in_channels=144, out_channels=96, kernel_size =3, stride=1, padding =1)
        self.dec2_1 = Conv2d_relu(in_channels=96, out_channels=96, kernel_size =3, stride=1, padding =1)
        self.dec1_2 = Conv2d_relu(in_channels=144, out_channels=96, kernel_size =3, stride=1, padding =1)
        self.dec1_1 = nn.Conv2d(in_channels=96, out_channels=3, kernel_size =3, stride=1, padding =1)
        self.pool = nn.MaxPool2d(kernel_size = 2)
        self.unpool = nn.Upsample(scale_factor=2, mode="nearest")
        
        with torch.no_grad():
            self._init_weights()
        
    def forward(self, x):
        enc1 = self.enc1(x)
        pool1 = self.pool(enc1)
        enc2 = self.enc2(pool1)
        pool2 = self.pool(enc2)
        enc3 = self.enc3(pool2)
        pool3 = self.pool(enc3)
        enc4 = self.enc4(pool3)
        pool4 = self.pool(enc4)
        enc5 = self.enc5(pool4)
        pool5 = self.pool(enc5)
        enc6 = self.enc6(pool5)
        
        unpool5 = self.unpool(enc6)
        cat5 = torch.cat((unpool5, enc5),dim =1)
        dec5_2 = self.dec5_2(cat5)
        dec5_1 = self.dec5_1(dec5_2)
        unpool4 = self.unpool(dec5_1)
        cat4 = torch.cat((unpool4, enc4),dim =1)
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)
        unpool3 = self.unpool(dec4_1)
        cat3 = torch.cat((unpool3, enc3),dim =1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)
        unpool2 = self.unpool(dec3_1)
        cat2 = torch.cat((unpool2, enc2),dim =1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)
        unpool1 = self.unpool(dec2_1)
        cat1 = torch.cat((unpool1, enc1),dim =1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)        
        return dec1_1

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, a=0.1)
                m.bias.data.zero_()