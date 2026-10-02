"""
Classic FCN-8s (Long et al., 2015) with a VGG-16 backbone.

FIXES
  * fc6 / fc7 are initialised from VGG-16's pretrained classifier weights
    (reshaped to 7x7 and 1x1 convolutions), exactly as in the FCN paper,
    instead of ~120M randomly initialised parameters.
  * The backbone lives under "backbone.*" so train.py gives it the same 0.1x
    encoder learning rate as every other model's encoder.
"""
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


class _VGGBackbone(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.to_pool3 = nn.Sequential(*features[:17])    # stride 8,  256 ch
        self.to_pool4 = nn.Sequential(*features[17:24])  # stride 16, 512 ch
        self.to_pool5 = nn.Sequential(*features[24:31])  # stride 32, 512 ch
        self.fc6 = nn.Conv2d(512, 4096, 7, padding=3)
        self.fc7 = nn.Conv2d(4096, 4096, 1)


class FCN8s(nn.Module):
    def __init__(self, num_classes, pretrained_backbone=True):
        super().__init__()
        weights = tvm.VGG16_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        vgg = tvm.vgg16(weights=weights)
        self.backbone = _VGGBackbone(list(vgg.features.children()))

        if pretrained_backbone:
            fc6, fc7 = vgg.classifier[0], vgg.classifier[3]
            self.backbone.fc6.weight.data.copy_(fc6.weight.data.view(4096, 512, 7, 7))
            self.backbone.fc6.bias.data.copy_(fc6.bias.data)
            self.backbone.fc7.weight.data.copy_(fc7.weight.data.view(4096, 4096, 1, 1))
            self.backbone.fc7.bias.data.copy_(fc7.bias.data)

        self.drop = nn.Dropout2d(0.5)
        self.score_fr = nn.Conv2d(4096, num_classes, 1)
        self.score_pool4 = nn.Conv2d(512, num_classes, 1)
        self.score_pool3 = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        H, W = x.shape[-2:]
        bb = self.backbone
        p3 = bb.to_pool3(x)
        p4 = bb.to_pool4(p3)
        p5 = bb.to_pool5(p4)
        h = self.drop(F.relu(bb.fc6(p5), inplace=True))
        h = self.drop(F.relu(bb.fc7(h), inplace=True))
        score = self.score_fr(h)
        score = F.interpolate(score, size=p4.shape[-2:], mode="bilinear", align_corners=False)
        score = score + self.score_pool4(p4)
        score = F.interpolate(score, size=p3.shape[-2:], mode="bilinear", align_corners=False)
        score = score + self.score_pool3(p3)
        return F.interpolate(score, size=(H, W), mode="bilinear", align_corners=False)
