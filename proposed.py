"""
A single configurable class that reproduces your proposed model
(DINOv2 -> FPN -> ASPP -> Residual Decoder -> composite loss) and can toggle
OFF any one component, so both ablation tables (Comment 5) come from the same
code path -- no separate re-implementation per variant, no risk of the
variants silently drifting apart.

  Cumulative table:            A -> B -> C -> D -> E (=Full/Proposed)
  Component-removal table:     Full, -DINOv2, -FPN, -ASPP, -ResidualDecoder
                                (-Composite Loss = ablation D, loss swapped in train.py)

Parameter names are IDENTICAL to your original StrongDINOv2Seg (encoder.*,
fpn.lat.*, fpn.fuse.*, aspp.*, dec1-4.*, seg_head.*, aux1-3.*), so the full
configuration loads your original best_model.pth with strict=True.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from transformers import AutoModel


# ─────────────────────────── building blocks ───────────────────────────────
class ResidualConvBlock(nn.Module):
    """The decoder block used by the FULL proposed model."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = (
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False),
                          nn.BatchNorm2d(out_ch))
            if in_ch != out_ch else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + self.skip(x))


class PlainConvBlock(nn.Module):
    """A minimal decoder block (no residual skip) used when the
    'Residual Decoder' component is ablated away."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch, rates=(6, 12, 18)):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.dilated = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=r, dilation=r, bias=False),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
            for r in rates])
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        n = 1 + len(rates) + 1
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * n, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Dropout2d(0.1))

    def forward(self, x):
        H, W = x.shape[-2:]
        out = [self.conv1(x)]
        for d in self.dilated:
            out.append(d(x))
        gp = F.interpolate(self.pool(x), size=(H, W), mode="bilinear", align_corners=False)
        out.append(gp)
        return self.project(torch.cat(out, dim=1))


class FPNFuse(nn.Module):
    """Generalized to accept per-level input channels, so it works with both
    the DINOv2 encoder (uniform 768-d hidden states) and a CNN encoder
    (256/512/1024/2048-d stages) for the '-DINOv2' ablation."""

    def __init__(self, in_channels_list, fpn_dim=256):
        super().__init__()
        self.lat = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, fpn_dim, 1, bias=False),
                nn.BatchNorm2d(fpn_dim), nn.ReLU(inplace=True))
            for c in in_channels_list])
        self.fuse = nn.Sequential(
            nn.Conv2d(fpn_dim * len(in_channels_list), fpn_dim * 2, 1, bias=False),
            nn.BatchNorm2d(fpn_dim * 2), nn.ReLU(inplace=True))

    def forward(self, feats):
        H, W = feats[0].shape[-2:]
        lats = []
        for i, f in enumerate(feats):
            l = self.lat[i](f)
            if l.shape[-2:] != (H, W):
                l = F.interpolate(l, size=(H, W), mode="bilinear", align_corners=False)
            lats.append(l)
        return self.fuse(torch.cat(lats, dim=1))


# ─────────────────────────── configurable model ────────────────────────────
class ConfigurableSegModel(nn.Module):
    """
    encoder_type          : "dinov2" (proposed) or "resnet50" (the '-DINOv2' ablation)
    use_fpn               : toggle the FPN fusion neck
    use_aspp              : toggle the ASPP module
    use_residual_decoder  : toggle residual (True) vs plain (False) decoder blocks

    Full/Proposed model   = ConfigurableSegModel("dinov2", True, True, True)
    """

    def __init__(self, num_classes, encoder_type="dinov2",
                 use_fpn=True, use_aspp=True, use_residual_decoder=True,
                 pretrained_cnn=True):
        super().__init__()
        self.encoder_type = encoder_type
        self.use_fpn = use_fpn
        self.use_aspp = use_aspp

        # ---- Encoder --------------------------------------------------
        if encoder_type == "dinov2":
            self.patch_size = 14
            self.encoder = AutoModel.from_pretrained(
                "facebook/dinov2-base", output_hidden_states=True)
            for p in self.encoder.parameters():
                p.requires_grad = True
            self.encoder_dims = [768, 768, 768, 768]
        elif encoder_type == "resnet50":
            weights = tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained_cnn else None
            resnet = tvm.resnet50(weights=weights)
            self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
            self.layer1 = resnet.layer1
            self.layer2 = resnet.layer2
            self.layer3 = resnet.layer3
            self.layer4 = resnet.layer4
            self.encoder_dims = [256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unknown encoder_type '{encoder_type}'")

        # ---- Neck (FPN) -------------------------------------------------
        if use_fpn:
            self.fpn = FPNFuse(self.encoder_dims, fpn_dim=256)
            neck_in_ch = 512   # fpn_dim * 2
        else:
            self.fpn = None
            self.reduce = nn.Sequential(
                nn.Conv2d(self.encoder_dims[-1], 512, 1, bias=False),
                nn.BatchNorm2d(512), nn.ReLU(inplace=True))
            neck_in_ch = 512

        # ---- Context (ASPP) ---------------------------------------------
        if use_aspp:
            self.aspp = ASPP(in_ch=neck_in_ch, out_ch=256)
        else:
            self.aspp = None
            self.neck_proj = nn.Sequential(
                nn.Conv2d(neck_in_ch, 256, 1, bias=False),
                nn.BatchNorm2d(256), nn.ReLU(inplace=True))

        # ---- Decoder ------------------------------------------------------
        Block = ResidualConvBlock if use_residual_decoder else PlainConvBlock
        self.up1, self.dec1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), Block(256, 256)
        self.up2, self.dec2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), Block(256, 128)
        self.up3, self.dec3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), Block(128, 64)
        self.up4, self.dec4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), Block(64, 32)

        self.seg_head = nn.Conv2d(32, num_classes, kernel_size=1)
        self.aux1 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.aux2 = nn.Conv2d(128, num_classes, kernel_size=1)
        self.aux3 = nn.Conv2d(64, num_classes, kernel_size=1)

    # ---- feature extraction -------------------------------------------
    def _vit_to_map(self, hidden, H, W):
        x = hidden[:, 1:, :]                 # drop CLS token
        B, N, C = x.shape
        Hp, Wp = H // self.patch_size, W // self.patch_size
        return x.permute(0, 2, 1).reshape(B, C, Hp, Wp)

    def _extract_feats(self, x):
        if self.encoder_type == "dinov2":
            H, W = x.shape[-2:]
            enc = self.encoder(pixel_values=x, output_hidden_states=True)
            layers = enc.hidden_states
            return [self._vit_to_map(layers[i], H, W) for i in (3, 6, 9, 12)]
        else:
            # FIX: resample every ResNet stage to the SAME 40x40 grid (stride 14)
            # that DINOv2 produces, so the '-DINOv2' row changes ONLY the encoder:
            # same neck resolution, same decoder output size, same compute budget
            # in the head. (Without this the decoder ran at 2240x2240.)
            H, W = x.shape[-2:]
            x0 = self.stem(x)
            c1 = self.layer1(x0)
            c2 = self.layer2(c1)
            c3 = self.layer3(c2)
            c4 = self.layer4(c3)
            size = (H // 14, W // 14)
            return [F.interpolate(c, size=size, mode="bilinear", align_corners=False)
                    for c in (c1, c2, c3, c4)]

    def forward(self, x):
        B, _, H, W = x.shape
        feats = self._extract_feats(x)

        neck_in = self.fpn(feats) if self.use_fpn else self.reduce(feats[-1])
        neck = self.aspp(neck_in) if self.use_aspp else self.neck_proj(neck_in)

        d1 = self.dec1(self.up1(neck))
        d2 = self.dec2(self.up2(d1))
        d3 = self.dec3(self.up3(d2))
        d4 = self.dec4(self.up4(d3))
        d4 = F.interpolate(d4, size=(H, W), mode="bilinear", align_corners=False)

        main = self.seg_head(d4)

        if self.training:
            a1 = F.interpolate(self.aux1(d1), size=(H, W), mode="bilinear", align_corners=False)
            a2 = F.interpolate(self.aux2(d2), size=(H, W), mode="bilinear", align_corners=False)
            a3 = F.interpolate(self.aux3(d3), size=(H, W), mode="bilinear", align_corners=False)
            return main, a1, a2, a3
        return main
