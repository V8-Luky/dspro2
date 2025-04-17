import cv2 as cv
import numpy as np
import torch.nn as nn
import torchvision.transforms.v2 as transforms

from PIL import Image


class RandomBackgroundTransform(nn.Module):
    def __init__(self, brightness=0.75, contrast=0.3, saturation=0.75, hue=0.4):
        super().__init__()
        self.color_jitter = transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def forward(self, img):
        transformed_foreground = np.array(self._apply_foreground_transform(img))

        img_np = np.array(img)
        hand_filter = self._get_hand_filter(img_np)

        transformed_img_with_background = self._apply_background(transformed_foreground, hand_filter)

        return Image.fromarray(transformed_img_with_background)

    def _apply_foreground_transform(self, img):
        return self.color_jitter(img)

    def _apply_background(self, img, mask):
        raise NotImplementedError("This method should be overridden in subclasses.")

    def _get_hand_filter(self, img):
        image_hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)

        hsv_min = [np.array([0, 0, 100])]
        hsv_max = [np.array([180, 60, 255])]

        hsv_min, hsv_max = np.array(hsv_min), np.array(hsv_max)

        background_image_hsv = sum([cv.inRange(image_hsv, lower, upper) for lower, upper in zip(hsv_min, hsv_max)])

        binary = np.copy(background_image_hsv)
        binary = cv.bitwise_not(binary)

        binary = cv.GaussianBlur(binary, (3, 3), 3)
        return binary


class RandomNoiseBackgroundTransform(RandomBackgroundTransform):
    def __init__(self, brightness=0.75, contrast=0.3, saturation=.75, hue=0.4):
        super().__init__(brightness, contrast, saturation, hue)

    def _apply_background(self, img, mask):
        img_with_noise = np.random.uniform(0, 255, size=img.shape).astype(np.uint8)
        img_with_noise[mask == 255] = img[mask == 255]
        img_with_noise = cv.GaussianBlur(img_with_noise, (3, 3), 1.5)
        return img_with_noise


class RandomRealLifeBackgroundTransform(RandomBackgroundTransform):
    def __init__(self, backgrounds: list[str], brightness=0.75, contrast=0.3, saturation=.75, hue=0.4):
        super().__init__(brightness, contrast, saturation, hue)
        self.backgrounds = [cv.imread(background) for background in backgrounds]

    def _apply_background(self, img, mask):
        background = np.random.choice(self.backgrounds)
        background = cv.resize(background, (img.shape[1], img.shape[0]))
        img_with_bg_replacement = background.copy()
        img_with_bg_replacement[mask == 255] = img[mask == 255]
        return img_with_bg_replacement
