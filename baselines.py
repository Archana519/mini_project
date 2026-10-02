"""
Standard segmentation architectures from segmentation-models-pytorch, all with
the SAME ResNet-34 ImageNet encoder so Comment 4 isolates decoder/head design.

FIX: U-Net / UNet++ / LinkNet require H and W divisible by 32, but the shared
image size is 560 (= 40 x 14 for DINOv2). PadToMultiple reflect-pads the input
to 576 and crops the logits back to 560, so every model still sees / is scored
on exactly the same 560x560 image and mask.
"""
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

_BUILDERS = {
    "unet": smp.Unet,
    "unetplusplus": smp.UnetPlusPlus,
    "pspnet": smp.PSPNet,
    "linknet": smp.Linknet,
    "deeplabv3plus": smp.DeepLabV3Plus,
}


class PadToMultiple(nn.Module):
    def __init__(self, model, multiple=32):
        super().__init__()
        self.model = model
        self.multiple = multiple

    def forward(self, x):
        H, W = x.shape[-2:]
        ph, pw = (-H) % self.multiple, (-W) % self.multiple
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        return self.model(x)[..., :H, :W]


def build_smp_baseline(name, num_classes, encoder_name="resnet34",
                       encoder_weights="imagenet"):
    key = name.lower()
    if key not in _BUILDERS:
        raise ValueError(f"Unknown smp baseline '{name}'. Available: {list(_BUILDERS)}")
    net = _BUILDERS[key](encoder_name=encoder_name, encoder_weights=encoder_weights,
                         in_channels=3, classes=num_classes)
    return PadToMultiple(net, 32)
