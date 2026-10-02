"""
TotalLoss     = CE + Dice + Boundary  -> the "composite loss" in your proposed model.
SimpleCELoss  = plain CE only         -> used by ablation rows A-D (D doubles as the "-Composite Loss" row).

Both losses transparently support models that return either a single logits
tensor, or a (main, aux1, aux2, aux3) tuple (deep supervision), so the SAME
loss classes work for baselines and for the proposed/ablation models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import cfg


class DiceLoss(nn.Module):
    def __init__(self, num_classes, ignore_index=0, smooth=1e-6):
        super().__init__()
        self.C = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits, targets):
        logits = logits.float()
        targets = targets.long()
        probs = F.softmax(logits, dim=1)
        B, C, H, W = probs.shape
        one_hot = (F.one_hot(targets.clamp(0, C - 1), C)
                   .permute(0, 3, 1, 2).float())
        total, count = 0.0, 0
        for c in range(C):
            if c == self.ignore_index:
                continue
            p = probs[:, c].reshape(B, -1)
            t = one_hot[:, c].reshape(B, -1)
            inter = (p * t).sum(dim=1)
            union = p.sum(dim=1) + t.sum(dim=1)
            total += (1.0 - (2 * inter + self.smooth) /
                      (union + self.smooth)).mean()
            count += 1
        return total / max(count, 1)


class BoundaryLoss(nn.Module):
    def __init__(self):
        super().__init__()
        sx = torch.tensor(
            [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        ).reshape(1, 1, 3, 3)
        sy = sx.permute(0, 1, 3, 2).contiguous()
        self.register_buffer("sobel_x", sx)
        self.register_buffer("sobel_y", sy)

    def forward(self, logits, targets):
        m = targets.unsqueeze(1).float()
        sx = self.sobel_x.to(dtype=torch.float32, device=m.device)
        sy = self.sobel_y.to(dtype=torch.float32, device=m.device)
        ex = F.conv2d(m, sx, padding=1)
        ey = F.conv2d(m, sy, padding=1)
        edge = (ex.abs() + ey.abs()).clamp(0, 1)
        weight = (1.0 + 4.0 * edge).squeeze(1)
        ce = F.cross_entropy(logits.float(), targets.long(),
                              ignore_index=0, reduction="none")
        return (ce * weight).mean()


class TotalLoss(nn.Module):
    """CE + Dice + Boundary, with optional deep-supervision on aux heads."""

    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=0)
        self.dice = DiceLoss(cfg.NUM_CLASSES, ignore_index=0, smooth=cfg.DICE_SMOOTH)
        self.bnd = BoundaryLoss()

    def _single(self, logits, targets):
        l = logits.float()
        t = targets.long()
        return (cfg.CE_W * self.ce(l, t) +
                cfg.DICE_W * self.dice(l, t) +
                cfg.EDGE_W * self.bnd(l, t))

    def forward(self, outputs, targets):
        if isinstance(outputs, (list, tuple)):
            main, a1, a2, a3 = outputs
            loss = self._single(main, targets)
            loss += cfg.AUX_W * self._single(a1, targets)
            loss += cfg.AUX_W * self._single(a2, targets)
            loss += cfg.AUX_W * self._single(a3, targets)
            return loss
        return self._single(outputs, targets)


class SimpleCELoss(nn.Module):
    """Plain cross-entropy only -- used by the '-Composite Loss' ablation."""

    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=0)

    def _single(self, logits, targets):
        return self.ce(logits.float(), targets.long())

    def forward(self, outputs, targets):
        if isinstance(outputs, (list, tuple)):
            main, a1, a2, a3 = outputs
            loss = self._single(main, targets)
            loss += cfg.AUX_W * self._single(a1, targets)
            loss += cfg.AUX_W * self._single(a2, targets)
            loss += cfg.AUX_W * self._single(a3, targets)
            return loss
        return self._single(outputs, targets)
