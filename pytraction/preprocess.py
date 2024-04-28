import numpy as np
import h5py
from typing import Tuple, Type, Union, Any

from shapely import geometry
from pytraction.utils import bead_density


def get_raw_frames(img_stack: np.ndarray, ref_stack: Union[np.ndarray, None], frame: int, bead_channel: int,
                   cell_channel: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract normalized bead frame from image and reference stack and get cell channel from image stack.
    """
    img = normalize(np.array(img_stack[frame, bead_channel, :, :]))
    if ref_stack is not None:
        ref = normalize(np.array(ref_stack[bead_channel, :, :]))
    else:
        ref = normalize(np.array(img_stack[frame + 1, bead_channel, :, :]))
    cell_img = normalize(np.array(img_stack[frame, cell_channel, :, :]))
    return img, ref, cell_img


def normalize(x: np.ndarray) -> np.ndarray:
    """
    Maps pixel intensity values to the interval [0, 255], corresponding to 8-bit images.
    """
    x = (x - np.min(x)) / (np.max(x) - np.min(x))  # Spreads the range to [0, 1]
    return np.array(x * 255, dtype="uint8")  # Array is explicitly cast to unsigned 8-bit integer


def get_min_window_size(img: np.ndarray, config) -> int:
    """
    Calculate window size from bead density if window_size is not set in pyforce config file.
    """
    if not config.config["piv"]["window_size"]:
        density = bead_density(img)

        if config.knn is None:
            wnd_sz = config.config["piv"]["window_size"]
            msg = f"Window size set to {wnd_sz}, but KNN model was not loaded."
            raise RuntimeWarning(msg)

        # Use K-nearest neighbors (KNN) classifier to predict minimum window size based on bead density
        knn = config.knn
        window_size = knn.predict([[density]])
        print(f"Automatically selected window size of {window_size}")
        return int(window_size)

    else:
        return config.config["piv"]["window_size"]
