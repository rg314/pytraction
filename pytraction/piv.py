import numpy as np
from typing import Tuple

from openpiv import pyprocess, validation, filters, tools
from pytraction.widim import WiDIM


def extended_area_piv(img: np.ndarray,
                      ref: np.ndarray,
                      config) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute deformation field using particle image velocimetry (PIV) with an implementation of extended search area.
    """
    config = config.config

    # Calculate window overlap in pixels
    overlap = int(config["piv"]["overlap_ratio"] * config["piv"]["window_size"])

    # Compute displacement field using a standard PIV cross-correlation algorithm
    u0, v0, sig2noise = pyprocess.extended_search_area_piv(
        frame_a=ref.astype(np.int32),
        frame_b=img.astype(np.int32),
        window_size=config["piv"]["window_size"],
        overlap=overlap,
        dt=config["piv"]["dt"],
        search_area_size=config["piv"]["window_size"],
        correlation_method=config["piv"]["correlation_method"],
        subpixel_method=config["piv"]["subpixel_method"],
        sig2noise_method=config["piv"]["sig2noise_method"],
        width=config["piv"]["width"],
        normalized_correlation=config["piv"]["normalized_correlation"],
        use_vectorized=config["piv"]["use_vectorized"]
    )

    # Return coordinates for PIV vector field
    x0, y0 = pyprocess.get_coordinates(image_size=ref.shape,
                                       search_area_size=config["piv"]["window_size"],
                                       overlap=overlap)

    # Remove vectors in field which have a signal-to-noise ratio larger than the threshold
    flags = validation.sig2noise_val(sig2noise, threshold=config["piv"]["sig2noise_threshold"])
    u_f, v_f = filters.replace_outliers(u0, v0, flags,
                                        method=config["piv"]["outliers_method"],
                                        max_iter=3,
                                        kernel_size=2)

    # Transform from image to physical coordinates
    x, y, u, v = tools.transform_coordinates(x0, y0, u_f, v_f)

    return x, y, u, v


def widim_piv(img: np.ndarray,
              ref: np.ndarray,
              config) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute deformation field using particle image velocimetry (PIV) with an implementation of window size coarsening.
    """
    config = config.config

    try:
        # Compute displacement field using a window deformation PIV algorithm
        x, y, u, v, mask = WiDIM(
            frame_a=ref.astype(np.int32),
            frame_b=img.astype(np.int32),
            mark=np.ones_like(ref).astype(np.int32),
            min_window_size=config["piv"]["window_size"],
            overlap_ratio=config["piv"]["overlap_ratio"],
            coarse_factor=config["piv"]["coarse_factor"],
            dt=config["piv"]["dt"],
            validation_method=config["piv"]["validation_method"],
            trust_1st_iter=config["piv"]["trust_1st_iter"],
            validation_iter=config["piv"]["validation_iter"],
            tolerance=config["piv"]["tolerance"],
            nb_iter_max=config["piv"]["nb_iter_max"],
            subpixel_method=config["piv"]["subpixel_method"],
            sig2noise_method=config["piv"]["sig2noise_method"],
            sig2noise_threshold=config["piv"]["sig2noise_threshold"],
            width=config["piv"]["width"]
        )
        return x, y, u, v

    except Exception as e:
        if isinstance(e, ZeroDivisionError):
            # Reduce window size and call compute_piv again
            config_tmp = config.copy()
            config_tmp["piv"]["window_size"] = (
                    config["piv"]["window_size"] // 2
            )
            print(f"Reduced min window size to {config_tmp['piv']['window_size'] // 2} in recursive call.")
            return widim_piv(img, ref, config_tmp)
        else:
            raise e
