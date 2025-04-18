import cv2 as cv
import numpy as np
import torch.nn as nn
import torchvision.transforms.v2 as transforms

from PIL import Image


def _get_mask(img, hsv_min, hsv_max):
    """Gets a mask for the background on the image."""
    image_hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    return cv.inRange(image_hsv, hsv_min, hsv_max)


class ExtractHand(nn.Module):
    """Extracts the hand from the image using a mask based on HSV color space.

    This transform should only be used on images obtained from the https://www.kaggle.com/datasets/kapillondhe/american-sign-language 
    dataset.
    """

    def forward(self, img):
        img_np = np.array(img)
        hand_filter = self._get_hand_filter(img_np)
        hand_only_image = self._apply_mask(img_np, hand_filter)
        return Image.fromarray(hand_only_image)

    def _get_hand_filter(self, img):
        """Gets a mask for the hand on the image."""
        bg_mask = _get_mask(img, np.array([0, 0, 100]), np.array([180, 60, 255]))
        hand_mask = cv.bitwise_not(bg_mask)
        return cv.GaussianBlur(hand_mask, (3, 3), 3)

    def _apply_mask(self, img, mask):
        """Applies a random noise background to the image based on the mask."""

        black = np.zeros_like(img)
        mask = np.expand_dims(mask, -1)

        hand_only_image = np.astype(img * (mask / 255), np.uint8)
        return hand_only_image


class RandomBackground(nn.Module):
    def __init__(self, background_filter_hsv: tuple[np.ndarray] = (np.array([0, 0, 0]), np.array([180, 5, 5])), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.background_filter_hsv = background_filter_hsv

    def forward(self, img):
        img_np = np.array(img)
        img_with_background = self._apply_background(img_np, self._get_background_filter(img_np) == 255)
        return Image.fromarray(img_with_background)

    def _apply_background(self, img, mask):
        raise NotImplementedError("This class is an abstract class and should not be used directly.")

    def _get_background_filter(self, img):
        """Gets a mask for the background on the image."""
        return _get_mask(img, self.background_filter_hsv[0], self.background_filter_hsv[1])


class RandomBackgroundNoise(RandomBackground):
    def _apply_background(self, img, mask):
        """Applies a random noise background to the image based on the mask."""

        noise = np.random.uniform(0, 255, size=img.shape).astype(np.uint8)
        img[mask] = noise[mask]
        return cv.GaussianBlur(img, (3, 3), 1.5)


class RandomRealLifeBackground(RandomBackground):

    def __init__(self, backgrounds: list[str], background_filter_hsv: tuple[np.ndarray] = (np.array([0, 0, 0]), np.array([180, 5, 5])), *args, **kwargs):
        super().__init__(background_filter_hsv, *args, **kwargs)
        self.backgrounds = [cv.cvtColor(cv.imread(background), cv.COLOR_BGR2RGB) for background in backgrounds]

    def _apply_background(self, img, mask):
        """Applies a random real-life background to the image based on the mask."""
        background_idx = np.random.randint(0, len(self.backgrounds))
        background = self.backgrounds[background_idx]

        background = cv.resize(background, (img.shape[0], img.shape[1]))
        img[mask] = background[mask]
        return cv.GaussianBlur(img, (3, 3), 1.5)
