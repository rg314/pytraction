from typing import Any
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt

from pytraction.net.dataloader import get_preprocessing


def get_mask(img: np.ndarray,
             model: Any,
             pre_fn: Any,
             device: str = "cuda") -> np.ndarray:
    """
    Calculates binary mask of cells using CNN feed-forward neural network.
    """
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # Convert COLOR_GRAY2BGR image to COLOR_BGR2LAB color space
    img = np.asarray(img)[:, :, :3]  # x, y, time-frame?
    image = pre_fn(image=img)["image"][:1, :, :]  # CNN preprocessing function
    # Predict mask from image using CNN model
    x_tensor = torch.from_numpy(image).to(device).unsqueeze(0)
    pr_mask = model.predict(x_tensor)

    return pr_mask.squeeze().cpu().numpy().round()  # Squeeze returns the tensor with all dimensions of size 1 removed
