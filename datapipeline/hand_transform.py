import torch
import torch.nn as nn
import numpy as np
import cv2 as cv
from PIL import Image

class HandSegmentationBatchTransform(nn.Module):
    def __init__(self, device="cpu"):
        self.device = device

    def get_hand_filter(img):
        image_hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
       
        hsv_min = [np.array([0, 0, 100])]
        hsv_max = [np.array([180, 60,255])]
        
        hsv_min, hsv_max = np.array(hsv_min), np.array(hsv_max)
           
        background_image_hsv = sum([cv.inRange(image_hsv, lower, upper) for lower, upper in zip(hsv_min, hsv_max)])
        np.min(skin_image_hsv), np.max(skin_image_hsv)
        
        binary = np.copy(skin_image_hsv)
        binary = cv.bitwise_not(binary)
        
        binary = cv.GaussianBlur(binary, (3,3), 3)
        return binary

    def __call__(self, img):
        """
        batch_tensor: Tensor of shape [B, C, H, W] in range [0, 1]
        returns: Tensor of shape [B, C, H, W] with hand isolated
        """

        img_rgb = np.array(img)
        
        # Get the mask
        mask = self.get_hand_filter(img_rgb)

        # Normalize and expand
        mask = mask.astype(np.float32) / 255.0
        mask = np.expand_dims(mask, axis=-1)

        # Apply to image
        masked_img = img_np.astype(np.float32) * mask

        return Image.fromarray(masked_img)
