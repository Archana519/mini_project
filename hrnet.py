"""
HRNet-W18 baseline: timm backbone + HRNetV2 segmentation head
(upsample all branches to the highest-resolution branch, concatenate,
1x1 conv -> BN -> ReLU -> 1x1 conv to classes).

FIX: timm's features_only HRNet can return an extra stride-2 stem level and
"incre" (widened) features. We request the raw stage-4 branch outputs
(feature_location="") and drop any level with stride < 4, so the head sees the
four HRNet branches at strides 4/8/16/32 as in the HRNetV2 paper.
The chosen levels are printed once at construction for your log.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


def _make_backbone(name, pretrained):
    try:
        return timm.create_model(name, pretrained=pretrained, features_only=True,
                                 feature_location="")
    except TypeError:            # older/newer timm without that kwarg
        return timm.create_model(name, pretrained=pretrained, features_only=True)


class HRNetSeg(nn.Module):
    def __init__(self, num_classes, backbone="hrnet_w18", pretrained=True, verbose=True):
        super().__init__()
        self.backbone = _make_backbone(backbone, pretrained)
        chs = self.backbone.feature_info.channels()
        reds = self.backbone.feature_info.reduction()
        self.keep = [i for i, r in enumerate(reds) if r >= 4]
        if verbose:
            print(f"[HRNet] timm levels: channels={chs} strides={reds} -> using {self.keep}")
        total_ch = sum(chs[i] for i in self.keep)
        self.head = nn.Sequential(
            nn.Conv2d(total_ch, 256, 1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, 1),
        )

    def forward(self, x):
        H, W = x.shape[-2:]
        feats = self.backbone(x)
        feats = [feats[i] for i in self.keep]
        size = feats[0].shape[-2:]
        ups = [feats[0]] + [F.interpolate(f, size=size, mode="bilinear", align_corners=False)
                            for f in feats[1:]]
        out = self.head(torch.cat(ups, dim=1))
        return F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
