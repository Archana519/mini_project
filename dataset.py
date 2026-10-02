"""
Dataset / preprocessing / augmentation -- byte-for-byte the same behaviour as
your original StrongDINOv2Seg script (resize to 560, /255, no mean/std
normalisation, h/v flip + rot90). Used by EVERY model.
"""
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from config import cfg


class LoveDADataset(Dataset):
    def __init__(self, image_paths, mask_paths, augment=False):
        assert len(image_paths) == len(mask_paths)
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        if image is None:
            raise FileNotFoundError(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(self.mask_paths[idx])

        image = cv2.resize(image, (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE))
        mask = cv2.resize(mask, (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE),
                          interpolation=cv2.INTER_NEAREST)

        if self.augment:
            if np.random.rand() > 0.5:
                image = np.fliplr(image).copy()
                mask = np.fliplr(mask).copy()
            if np.random.rand() > 0.5:
                image = np.flipud(image).copy()
                mask = np.flipud(mask).copy()
            k = np.random.randint(0, 4)
            image = np.rot90(image, k).copy()
            mask = np.rot90(mask, k).copy()

        image = image.astype(np.float32) / 255.0
        image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)
        mask = torch.tensor(mask.astype(np.int64), dtype=torch.long)
        mask = mask.clamp(0, cfg.NUM_CLASSES - 1)
        return image, mask


def collect_split(split_root, domains=None):
    domains = domains or cfg.DOMAINS
    images, masks = [], []
    for domain in domains:
        img_dir = os.path.join(split_root, domain, "images_png")
        mask_dir = os.path.join(split_root, domain, "masks_png")
        for f in sorted(os.listdir(img_dir)):
            images.append(os.path.join(img_dir, f))
            masks.append(os.path.join(mask_dir, f))
    return images, masks


def get_splits():
    tr = collect_split(os.path.join(cfg.DATA_ROOT, "Train", "Train"))
    va = collect_split(os.path.join(cfg.DATA_ROOT, "Val", "Val"))
    return tr, va
