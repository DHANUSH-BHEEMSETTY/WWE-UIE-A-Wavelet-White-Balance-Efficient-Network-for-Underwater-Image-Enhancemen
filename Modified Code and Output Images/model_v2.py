import torch
import torch.nn as nn
import torch.nn.functional as F






class SepConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1,
                 bias=True, padding_mode="zeros"):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channel, in_channel, kernel_size,
            stride=stride, padding=kernel_size // 2,
            groups=in_channel, bias=bias, padding_mode=padding_mode,
        )
        self.conv2 = nn.Conv2d(in_channel, out_channel, kernel_size=1,
                               stride=1, padding=0, bias=bias)

    def forward(self, x):
        return self.conv2(self.conv1(x))


class BasicBlock(nn.Module):
    """HIN Block – identical to baseline."""
    def __init__(self, in_size, out_size, kernel_size=3, relu_slope=0.1):
        super().__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.conv_1 = SepConv(in_size, out_size, kernel_size=kernel_size)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=True)
        self.conv_2 = SepConv(out_size, out_size, kernel_size=kernel_size)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=True)
        self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)

    def forward(self, x):
        out = self.conv_1(x)
        out_1, out_2 = torch.chunk(out, 2, dim=1)
        out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))
        return out + self.identity(x)


class WaveletEnhanceBlock(nn.Module):
    """Haar wavelet sub-band decompose + fuse – identical to baseline."""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        ll = torch.tensor([[0.5,  0.5], [ 0.5,  0.5]])
        lh = torch.tensor([[-0.5,-0.5], [ 0.5,  0.5]])
        hl = torch.tensor([[-0.5, 0.5], [-0.5,  0.5]])
        hh = torch.tensor([[ 0.5,-0.5], [-0.5,  0.5]])
        kernel = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        kernel = kernel.repeat(channels, 1, 1, 1)
        self.register_buffer("haar_kernel", kernel)
        self.fuse = nn.Conv2d(4 * channels, channels, kernel_size=1, bias=False)
        self.post = SepConv(channels, channels, kernel_size=3, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        dwt  = F.conv2d(x, self.haar_kernel, stride=2, groups=C)
        fea  = self.post(self.fuse(dwt))
        return F.interpolate(fea, size=(H, W), mode="bilinear", align_corners=False)


class GetGradient(nn.Module):
    def __init__(self, dim=3, mode="sobel"):
        super().__init__()
        self.dim, self.mode = dim, mode
        if mode == "sobel":
            ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            self.register_buffer("kernel_y", ky.repeat(dim, 1, 1, 1))
            self.register_buffer("kernel_x", kx.repeat(dim, 1, 1, 1))

    def forward(self, x):
        gx = F.conv2d(x, self.kernel_x, padding=1, groups=self.dim)
        gy = F.conv2d(x, self.kernel_y, padding=1, groups=self.dim)
        return torch.sqrt(gx**2 + gy**2 + 1e-6)


class SGFB(nn.Module):
    def __init__(self, feature_channels=48):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.frdb1 = BasicBlock(feature_channels, feature_channels, kernel_size=3)
        self.frdb2 = BasicBlock(feature_channels, feature_channels, kernel_size=3)
        self.get_gradient = GetGradient(feature_channels, mode="sobel")
        self.conv_grad = nn.Sequential(
            SepConv(feature_channels, feature_channels, kernel_size=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        grad = self.conv_grad(self.get_gradient(x))
        x = self.frdb1(x)
        alpha = torch.sigmoid(self.alpha)
        x = alpha * grad * x + (1 - alpha) * x
        return self.frdb2(x)






class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation channel attention.
    Learns channel-wise importance weights → recalibrates features.
    Ratio=4 keeps overhead tiny (~1% extra params per BasicLayer).
    """
    def __init__(self, channels, ratio=4):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // ratio, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.se(x).view(x.size(0), x.size(1), 1, 1)
        return x * w






class BasicLayer(nn.Module):
    """
    Upgraded processing block:
      WaveletEnhanceBlock → SE Channel Attention → SGFB
    Compared to baseline: added SEBlock between wavelet and gradient steps.
    """
    def __init__(self, feature_channels=48):
        super().__init__()
        self.fwawb = WaveletEnhanceBlock(feature_channels)
        self.se    = SEBlock(feature_channels, ratio=4)          
        self.sgfb  = SGFB(feature_channels)

    def forward(self, x):
        res = x
        x   = self.se(self.fwawb(x) + x)                        
        x   = self.sgfb(x)
        return 0.5 * x + 0.5 * res






class Downsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, 3, 1, 1, bias=False),
            nn.PixelUnshuffle(2),
        )
    def forward(self, x): return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, 3, 1, 1, bias=False),
            nn.PixelShuffle(2),
        )
    def forward(self, x): return self.body(x)






class GrayWorldRetinex(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=(2, 3), keepdim=True)
        gray_mean = mean.mean(dim=1, keepdim=True)
        gain = gray_mean / (mean + self.eps)
        x = x * gain
        x_log = torch.log(x + self.eps)
        x_log = x_log - x_log.mean(dim=(2, 3), keepdim=True)
        x_out = torch.exp(x_log)
        x_min = x_out.amin(dim=(-2, -1), keepdim=True)
        x_max = x_out.amax(dim=(-2, -1), keepdim=True)
        return (x_out - x_min) / (x_max - x_min + self.eps)










class myModelV2(nn.Module):
    """
    Enhanced underwater image enhancement model.

    Key differences from baseline myModel:
      - feature_channels: 32 → 48 (wider, ~65% more capacity)
      - Depth: 2-level → 3-level U-Net (larger receptive field for color casts)
      - SEBlock channel attention inside every BasicLayer
    """
    def __init__(self, in_channels=3, feature_channels=48, use_white_balance=True):
        super().__init__()
        C = feature_channels

        self.use_white_balance = use_white_balance
        if use_white_balance:
            self.wb    = GrayWorldRetinex()
            self.alpha = nn.Parameter(torch.zeros(1, 3, 1, 1), requires_grad=True)

        
        self.first = nn.Conv2d(in_channels, C, 3, 1, 1)

        
        self.encoder1 = BasicLayer(C)           
        self.down1    = Downsample(C)            

        self.encoder2 = BasicLayer(C * 2)       
        self.down2    = Downsample(C * 2)        

        self.encoder3 = BasicLayer(C * 4)       
        self.down3    = Downsample(C * 4)        

        
        self.bottleneck = BasicLayer(C * 8)     

        
        self.up3      = Upsample(C * 8)          
        self.decoder3 = BasicLayer(C * 4)        
        self.up2      = Upsample(C * 4)          
        self.decoder2 = BasicLayer(C * 2)
        self.up1      = Upsample(C * 2)          
        self.decoder1 = BasicLayer(C)

        
        self.out = nn.Conv2d(C, in_channels, 3, 1, 1)

    def forward(self, x):
        res = x

        
        if self.use_white_balance:
            alpha = torch.sigmoid(self.alpha)
            x = alpha * self.wb(x) + (1 - alpha) * x

        
        x1 = self.encoder1(self.first(x))   

        
        x2 = self.encoder2(self.down1(x1))  
        x3 = self.encoder3(self.down2(x2))  

        
        xb = self.bottleneck(self.down3(x3))  

        
        x  = self.decoder3(self.up3(xb) + x3)   
        x  = self.decoder2(self.up2(x)  + x2)   
        x  = self.decoder1(self.up1(x)  + x1)   

        return self.out(x) + res





if __name__ == "__main__":
    from thop import profile, clever_format

    dummy = torch.rand(1, 3, 256, 256)
    model = myModelV2()
    out   = model(dummy)
    print("Input  shape:", dummy.shape)
    print("Output shape:", out.shape)

    flops, params = profile(model, inputs=(dummy,), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")
    print(f"Params: {params}, FLOPs: {flops}")
