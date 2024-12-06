"""Helper functions for preprocessing retro gym rgb matrices"""

import numpy as np
import torch


def rgb2gray_luminance(frame: np.ndarray) -> np.ndarray:
    """
    Convert an RGB frame to grayscale using the standard luminance formula:
      Gray = 0.299 * R + 0.587 * G + 0.114 * B

    frame: A NumPy array of shape (H, W, 3) in HWC format
    returns: A NumPy array of shape (H, W), representing the grayscale image
    """
    # Ensure the input has the right shape
    if frame.shape[-1] != 3:
        raise ValueError(f"Expected last dimension = 3 for RGB, got {frame.shape}")

    # Apply the luminance formula
    # frame[..., 0] = R, frame[..., 1] = G, frame[..., 2] = B
    return 0.299 * frame[..., 0] + 0.587 * frame[..., 1] + 0.114 * frame[..., 2]


def stack_frames_grayscale(frames):
    """
    Convert each of 4 frames (H, W, 3) to grayscale, then stack them
    along the channel dimension for use with a CNN.

    frames: List of 4 NumPy arrays, each shape (H, W, 3) in HWC format.
    returns: A PyTorch tensor of shape (4, H, W) in CHW format,
             i.e., 4 grayscale channels.
    """
    stacked_list = []
    for f in frames:
        f_torch = torch.from_numpy(f).float().unsqueeze(0)
        stacked_list.append(f_torch)

    # Concatenate along channel dimension => (4, H, W)
    stacked_tensor = torch.cat(stacked_list, dim=0)
    return stacked_tensor


if __name__ == "__main__":
    # Example usage with dummy frames
    H, W = 200, 256
    frame1 = np.random.rand(H, W, 3).astype(np.float32)
    frame2 = np.random.rand(H, W, 3).astype(np.float32)
    frame3 = np.random.rand(H, W, 3).astype(np.float32)
    frame4 = np.random.rand(H, W, 3).astype(np.float32)

    frames = [frame1, frame2, frame3, frame4]
    state = stack_frames_grayscale(frames)
    print("Stacked shape (CHW):", state.shape)  # (4, H, W)
