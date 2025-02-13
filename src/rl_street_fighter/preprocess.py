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

    resized = cv2.resize(equalized, target_size, interpolation=cv2.INTER_AREA)

    # Normalize the pixel values to range [0, 1]
    normalized = resized.astype(np.float32) / 255.0

    # Convert to float16 to reduce memory usage
    processed = normalized.astype(np.float16)

    # Resize to target size

    return processed


def stack_frames(frames):
    stacked_list = []
    for f in frames:
        f_torch = torch.from_numpy(f).float().unsqueeze(0)
        stacked_list.append(f_torch)

    # Concatenate along channel dimension => (4, H, W)
    stacked_tensor = torch.cat(stacked_list, dim=0)
    return stacked_tensor
