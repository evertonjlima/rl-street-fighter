"""Helper functions for preprocessing retro gym rgb matrices"""

import cv2
import numpy as np
import torch


def preprocess_image(rgb_array, target_size=(96, 96)):
    # Convert RGB to Grayscale
    grayscale = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)

    # Ensure image is uint8 (0-255 range)
    grayscale = np.clip(grayscale, 0, 255).astype(np.uint8)

    # Enhance contrast using Histogram Equalization
    equalized = cv2.equalizeHist(grayscale)

    # Resize to target size
    resized = cv2.resize(equalized, target_size, interpolation=cv2.INTER_AREA)

    return resized


def stack_frames(frames):
    stacked_list = []
    for f in frames:
        f_torch = torch.from_numpy(f).float().unsqueeze(0)
        stacked_list.append(f_torch)

    # Concatenate along channel dimension => (4, H, W)
    stacked_tensor = torch.cat(stacked_list, dim=0)
    return stacked_tensor
