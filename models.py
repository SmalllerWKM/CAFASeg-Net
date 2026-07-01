import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1,
                 dilation=1, groups=1, act=True, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn  = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=1, bias=False):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, padding=dilation, dilation=dilation,
                            groups=in_ch, bias=bias)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return self.bn(self.pw(self.dw(x)))


class DSBABone(nn.Module):


    out_channels: Tuple[int, ...] = (64, 128, 256, 512)

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        from torchvision.models import resnet34, ResNet34_Weights

        rn = resnet34(weights=ResNet34_Weights.DEFAULT if pretrained else None)


        self.stem   = nn.Sequential(rn.conv1, rn.bn1, rn.relu, rn.maxpool)
        self.layer1 = rn.layer1
        self.layer2 = rn.layer2
        self.layer3 = rn.layer3
        self.layer4 = rn.layer4


        self.bnd_stem   = self._bnd_block(3,   32,  stride=4)
        self.bnd_layer1 = self._bnd_block(32,  64,  stride=1)
        self.bnd_layer2 = self._bnd_block(64,  128, stride=2)
        self.bnd_layer3 = self._bnd_block(128, 256, stride=2)
        self.bnd_layer4 = self._bnd_block(256, 512, stride=2)


        self.fuse1 = self._fuse_gate(64)
        self.fuse2 = self._fuse_gate(128)
        self.fuse3 = self._fuse_gate(256)
        self.fuse4 = self._fuse_gate(512)

        if pretrained:
            print("[DSBABone] Loaded pretrained ResNet34 weights for the semantic stream")

    @staticmethod
    def _bnd_block(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:

        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    @staticmethod
    def _fuse_gate(ch: int) -> nn.Sequential:

        return nn.Sequential(
            nn.Conv2d(ch * 2, ch, 1, bias=False),
            nn.BatchNorm2d(ch),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:

        s0 = self.stem(x)
        s1 = self.layer1(s0)
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        s4 = self.layer4(s3)


        b1 = self.bnd_layer1(self.bnd_stem(x))
        b2 = self.bnd_layer2(b1)
        b3 = self.bnd_layer3(b2)
        b4 = self.bnd_layer4(b3)


        g1 = self.fuse1(torch.cat([s1, b1], dim=1))
        g2 = self.fuse2(torch.cat([s2, b2], dim=1))
        g3 = self.fuse3(torch.cat([s3, b3], dim=1))
        g4 = self.fuse4(torch.cat([s4, b4], dim=1))

        return (
            g1 * s1 + (1.0 - g1) * b1,
            g2 * s2 + (1.0 - g2) * b2,
            g3 * s3 + (1.0 - g3) * b3,
            g4 * s4 + (1.0 - g4) * b4,
        )


class ASPPLite(nn.Module):
    def __init__(self, channels: int, drop_rate: float = 0.1) -> None:
        super().__init__()
        mid = max(channels // 4, 16)

        self.branch1 = nn.Sequential(
            nn.Conv2d(channels, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid), nn.GELU(),
        )
        self.branch6 = nn.Sequential(
            nn.Conv2d(channels, mid, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(mid), nn.GELU(),
        )
        self.branch12 = nn.Sequential(
            nn.Conv2d(channels, mid, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(mid), nn.GELU(),
        )
        self.branch_gp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.BatchNorm2d(mid), nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(mid * 4, channels, 1, bias=False),
            nn.BatchNorm2d(channels), nn.GELU(),
            nn.Dropout2d(p=drop_rate),
        )
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        b1   = self.branch1(x)
        b6   = self.branch6(x)
        b12  = self.branch12(x)
        bgp  = self.branch_gp(x).expand(-1, -1, H, W)
        fused = self.fuse(torch.cat([b1, b6, b12, bgp], dim=1))
        return self.norm(fused + x)


class CAFM(nn.Module):
    def __init__(self, enc_ch: int, dec_ch: int, out_ch: int, reduction: int = 4) -> None:
        super().__init__()
        self.align_enc = ConvBNAct(enc_ch, out_ch, 1, padding=0)
        self.align_dec = ConvBNAct(dec_ch, out_ch, 1, padding=0)

        mid_ch = max(1, out_ch // reduction)
        self.ch_mlp = nn.Sequential(
            nn.Linear(out_ch, mid_ch, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_ch, out_ch, bias=False),
            nn.Sigmoid(),
        )
        self.sp_conv = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.gate_conv = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch // 2, 1, bias=False),
            nn.BatchNorm2d(out_ch // 2),
            nn.GELU(),
            nn.Conv2d(out_ch // 2, 1, 3, padding=1, bias=False),
            nn.Sigmoid(),
        )
        self.refine = ConvBNAct(out_ch, out_ch, 3)

    def forward(self, f_enc: Tensor, f_dec: Tensor) -> Tensor:
        f_enc = self.align_enc(f_enc)
        f_dec = self.align_dec(f_dec)
        B, C, H, W = f_enc.shape


        m_ch = self.ch_mlp(f_enc.mean((2,3)) + f_enc.amax((2,3))).view(B, C, 1, 1)
        enc_att = f_enc * m_ch


        sp_feat  = torch.cat([f_dec.max(1, keepdim=True).values,
                               f_dec.mean(1, keepdim=True)], dim=1)
        dec_att  = f_dec * self.sp_conv(sp_feat)


        gate   = self.gate_conv(torch.cat([enc_att, dec_att], dim=1))
        fused  = gate * enc_att + (1.0 - gate) * dec_att
        return self.refine(fused)


class BERD(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels


        sx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        sy = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        self.Wx       = nn.Parameter(sx.repeat(channels, 1, 1, 1))
        self.Wy       = nn.Parameter(sy.repeat(channels, 1, 1, 1))
        self.sobel_bn = nn.BatchNorm2d(channels)


        self.struct_pool = nn.AvgPool2d(3, stride=1, padding=1)
        self.struct_bn   = nn.BatchNorm2d(channels)


        self.center_pool   = nn.AvgPool2d(3, stride=1, padding=1)
        self.surround_pool = nn.AvgPool2d(7, stride=1, padding=3)
        self.contrast_bn   = nn.BatchNorm2d(channels)


        mid = max(channels // 4, 8)
        self.fuse_mlp = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(mid, 3, 1, bias=False),
        )


        self.gamma     = nn.Parameter(torch.tensor(0.1))
        self.gate_conv = nn.Conv2d(channels, channels, 1, bias=False)
        self.out_conv  = ConvBNAct(channels, channels, 3)

    def forward(self, x: Tensor) -> Tensor:
        C = self.channels


        x_f32 = x.float()
        Wx_f32 = self.Wx.float()
        Wy_f32 = self.Wy.float()


        gx = F.conv2d(x_f32, Wx_f32, padding=1, groups=C)
        gy = F.conv2d(x_f32, Wy_f32, padding=1, groups=C)
        f1 = self.sobel_bn(torch.sqrt(gx ** 2 + gy ** 2 + 1e-8).to(x.dtype))


        Jxx = self.struct_pool(gx * gx)
        Jyy = self.struct_pool(gy * gy)
        Jxy = self.struct_pool(gx * gy)

        trace     = Jxx + Jyy
        det_delta = torch.sqrt(((Jxx - Jyy) * 0.5) ** 2 + Jxy ** 2 + 1e-8)
        lam1 = trace * 0.5 + det_delta
        lam2 = (trace * 0.5 - det_delta).clamp(min=0.)

        f2 = self.struct_bn(((lam1 - lam2) / (lam1 + lam2 + 1e-8)).to(x.dtype))


        f3 = self.contrast_bn(
            torch.abs(self.center_pool(x) - self.surround_pool(x))
        )


        w  = F.softmax(self.fuse_mlp(x), dim=1)
        w1 = w[:, 0:1]
        w2 = w[:, 1:2]
        w3 = w[:, 2:3]
        fb = w1 * f1 + w2 * f2 + w3 * f3


        gate = torch.sigmoid(self.gate_conv(x))
        return self.out_conv(x + self.gamma * (fb * gate))


class DecoderBlock(nn.Module):
    def __init__(self, enc_ch, dec_ch, out_ch):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.cafm = CAFM(enc_ch=enc_ch, dec_ch=dec_ch, out_ch=out_ch)

    def forward(self, f_dec, f_enc):
        return self.cafm(f_enc, self.up(f_dec))


class CAFASegNet(nn.Module):


    def __init__(
        self,
        in_channels:      int            = 3,
        encoder_channels: Tuple[int,...] = (64, 128, 256, 512),
        decoder_channels: Tuple[int,...] = (128, 64, 32, 16),
        num_classes:      int            = 1,
    ) -> None:
        super().__init__()
        e = encoder_channels
        d = decoder_channels

        self.backbone   = DSBABone(pretrained=True)
        self.bottleneck = ASPPLite(channels=e[3])
        self.dec3       = DecoderBlock(enc_ch=e[2], dec_ch=e[3], out_ch=d[0])
        self.dec2       = DecoderBlock(enc_ch=e[1], dec_ch=d[0], out_ch=d[1])
        self.dec1       = DecoderBlock(enc_ch=e[0], dec_ch=d[1], out_ch=d[2])

        self.dec0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            ConvBNAct(d[2], d[2], 3),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            ConvBNAct(d[2], d[3], 3),
        )
        self.aux_head3 = self._make_aux(d[0], num_classes)
        self.aux_head2 = self._make_aux(d[1], num_classes)
        self.berd      = BERD(d[3])
        self.head      = nn.Conv2d(d[3], num_classes, 1)

        self._init_non_backbone_weights()

    @staticmethod
    def _make_aux(in_ch: int, num_classes: int) -> nn.Sequential:
        mid = max(in_ch // 4, 16)
        return nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid, num_classes, 1),
        )

    def _init_non_backbone_weights(self) -> None:

        for name, m in self.named_modules():
            if name.startswith("backbone"):
                continue
            if isinstance(m, nn.Conv2d):
                if m.weight.shape[-1] == 3 and m.groups == m.in_channels:
                    continue
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def set_tau(self, tau: float) -> None:

        pass

    def forward(self, x: Tensor, return_features: bool = False) -> Dict[str, Tensor]:
        f1, f2, f3, f4 = self.backbone(x)
        fb = self.bottleneck(f4)
        d3 = self.dec3(fb, f3)
        d2 = self.dec2(d3, f2)
        d1 = self.dec1(d2, f1)
        d0 = self.dec0(d1)
        fr = self.berd(d0)
        lg = self.head(fr)

        out: Dict[str, Tensor] = {
            "logits": lg,
            "pred":   torch.sigmoid(lg),
        }
        if self.training:
            out["aux_logits3"] = self.aux_head3(d3)
            out["aux_logits2"] = self.aux_head2(d2)
        return out


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = CAFASegNet(decoder_channels=(128, 64, 32, 16)).to(device)
    x      = torch.randn(2, 3, 512, 512, device=device)
    model.train()
    out = model(x)
    print("logits:", out["logits"].shape)
    print("aux3:  ", out["aux_logits3"].shape)
    print("aux2:  ", out["aux_logits2"].shape)
    total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {total:.2f}M")
    print("Self-test passed.")
