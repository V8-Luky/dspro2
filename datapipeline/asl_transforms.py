import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
from torchvision.io import decode_image

from PIL import Image

import kornia
import kornia.filters as KF
import torch.nn.functional as F
import kornia.geometry.transform as KGT


class ExtractHand(nn.Module):
    def __init__(self):
        super().__init__()
        self.hsv_min = torch.tensor([0.0, 0.0, 0.4])[:, None, None]
        self.hsv_max = torch.tensor([2 * kornia.pi, 0.3, 1.0])[:, None, None]

    def forward(self, img):
        img_hsv = kornia.color.rgb_to_hsv(img)
        mask = ((img_hsv >= self.hsv_min.to(img.device)) &
                (img_hsv <= self.hsv_max.to(img.device))).all(dim=0, keepdim=True).float()

        hand_mask = 1.0 - mask

        return img * hand_mask


class RandomBackgroundBase(nn.Module):
    def __init__(self, hsv_min, hsv_max):
        super().__init__()
        self.hsv_min = hsv_min[:, None, None]
        self.hsv_max = hsv_max[:, None, None]

    def _get_background_mask(self, img):
        hsv = kornia.color.rgb_to_hsv(img)
        mask = ((hsv >= self.hsv_min.to(img.device)) &
                (hsv <= self.hsv_max.to(img.device))).all(dim=0, keepdim=True).float()
        return mask


class RandomBackgroundNoise(RandomBackgroundBase):
    def __init__(self):
        super().__init__(
            hsv_min=torch.tensor([0.0, 0.0, 0.0]),
            hsv_max=torch.tensor([2 * kornia.pi, 0.02, 0.02])
        )

    def forward(self, img):
        mask = self._get_background_mask(img)

        noise = torch.rand_like(img)
        return img * (1.0 - mask) + noise * mask


class RandomRealLifeBackground(RandomBackgroundBase):
    def __init__(self, backgrounds: list[str]):
        super().__init__(
            hsv_min=torch.tensor([0.0, 0.0, 0.0]),
            hsv_max=torch.tensor([2 * kornia.pi, 0.02, 0.02])
        )
        self.backgrounds = [decode_image(bg, mode="RGB").float() / 255.0 for bg in backgrounds]

    def forward(self, img):
        mask = self._get_background_mask(img)

        idx = torch.randint(0, len(self.backgrounds), (1,), device=img.device)
        selected_bg = self.backgrounds[idx]
        selected_bg = KGT.resize(selected_bg, img.shape[-2:])

        return img * (1.0 - mask) + selected_bg * mask
