import os
import zipfile
import pandas as pd
import numpy as np
import cv2

from shapely import geometry
from read_roi import read_roi_file
from typing import Tuple, Union, Any
from scipy.spatial import distance
from pytraction.net import segment as pynet


def roi_loaders(roi_path: Union[str, None]):
    """
    Call appropriate import functions depending on ROI file format.

    @param  roi_path: System path to file containing ROI information (.csv, .roi or .zip). No ROI will be loaded if set
    to None.
    """
    if roi_path is None:
        return None
    elif ".roi" in roi_path:
        return _load_roi(roi_path)
    elif ".csv" in roi_path:
        return _load_csv_roi(roi_path)
    elif ".zip" in roi_path:
        return _load_zip_roi(roi_path)
    else:
        msg = "The roi loader expects files in '.csv', '.roi', or '.zip' format"
        raise NotImplementedError(msg)


def _load_csv_roi(roi_path: str) -> Tuple[list, list]:
    """
    Load ROI data from .csv file containing two columns specifying the x and y coordinates.

    @param  roi_path: System path to .csv file containing ROI information.
    """
    x, y = pd.read_csv(roi_path).T.values
    return list(x), list(y)


def _load_roi(roi_path: str) -> Tuple[list, list]:
    """
    Load ROI data from .roi file.

    @param  roi_path: System path to .roi file containing ROI information.
    """
    d = read_roi_file(roi_path)

    # Return x and y coordinates from dictionary
    x = _recursive_lookup("x", d)
    y = _recursive_lookup("y", d)

    return x, y


def _load_zip_roi(roi_path: str) -> list:
    """
    Load ROI data from .zip folder.

    @param  roi_path: System path to .zip folder containing ROI information.
    """
    rois = []
    # Open .zip folder and extract each sub-file
    with zipfile.ZipFile(roi_path) as zip_roi:
        for file in zip_roi.namelist():
            roi_path_file = zip_roi.extract(file)

            d = read_roi_file(roi_path_file)

            # Return x and y coordinates from dictionary
            x = _recursive_lookup("x", d)
            y = _recursive_lookup("y", d)

            # Append x and y coordinates to array
            rois.append((x, y))

            # Delete extracted file
            os.remove(roi_path_file)
    return rois


def _recursive_lookup(key: str, dictionary: dict) -> Union[list, None]:
    """
    Given a nested dictionary d, return the first instance of d[k].

    @param  key: Target key.
    @param  dictionary: Nested dictionary.
    """
    if key in dictionary:
        return dictionary[key]
    for v in dictionary.values():
        if isinstance(v, dict):
            a = _recursive_lookup(key, v)
            if a is not None:
                return a
    return None


def load_frame_roi(roi, frame: int, nframes: int) -> Union[list, None]:
    """
    Load ROI for current frame.
    """
    if isinstance(roi, list):
        assert (
            len(roi) == nframes
        ), f"Warning ROI list has len {len(roi)} which is not equal to the number of frames ({nframes}). This would \
        suggest that you do not have the correct number of ROIs in the zip file."
        return roi[frame]  # Return ROI corresponding to active frame
    else:
        return roi  # Return single ROI (or None) which should be applied to all images


def get_polygon_and_roi(cell_img: np.ndarray,
                        roi: list,
                        config
                        ) -> Union[type(geometry.Polygon), type(np.ndarray)]:
    """
    Returns polygon shape of cell outline from CNN model prediction or ROI definition.
    """
    if roi:
        # Get x, y coordinates from ROI
        polyx, polyy = roi[0], roi[1]
    elif config.config["settings"]["segment"]:
        # Segments cell using CNN model and return x, y coordinates for shape
        polyx, polyy = _predict_roi(cell_img, config)  # Predict coordinates of cell outline
    else:
        return None, None

    pts = np.array(list(zip(polyx, polyy)), np.int32)  # Creates array of (x,y) coordinate pairs
    polygon = geometry.Polygon(pts)  # Create polygon shape from points
    pts = pts.reshape((-1, 1, 2))  # Get array with coordinate pairs
    return polygon, pts


def _predict_roi(cell_img: np.ndarray, config) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predicts the cell contour in an image using a CNN feed forward network and returns its coordinates.
    """
    # Predict boolean mask for cell detection
    mask = pynet.get_mask(cell_img, config.cnn, config.pre_fn, device=config.config["settings"]["device"])
    mask = np.array(mask.astype("bool"), dtype="uint8")

    # Detecting contours in mask using a simple chain approximation
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find the most central contour
    cntr_contour = _located_most_central_cell(contours, mask)

    # Extract x and y coordinates of central contour
    polyx, polyy = np.squeeze(cntr_contour, axis=1).T
    return polyx, polyy


def _located_most_central_cell(contours: Any, mask: np.ndarray) -> np.ndarray:
    """
    Calculates the distances of all contour centres to image centre and returns the most central one.
    """
    image_center = np.asarray(mask.shape) / 2
    image_center = tuple(image_center.astype("int32"))  # Convert centre coordinates to tuple

    segmented = []
    for contour in contours:
        mom = cv2.moments(contour)  # Calculate the moments of a shape up to the 3rd order

        center_x = int(mom["m10"] / mom["m00"])  # Divide 1st by 0th moment in the x-direction
        center_y = int(mom["m01"] / mom["m00"])  # Divide 1st by 0th moment in the y-direction
        contour_center = (center_x, center_y)

        # Calculate distance of contour centre to image centre
        distances_to_center = distance.euclidean(image_center, contour_center)

        segmented.append(
            {
                "contour": contour,
                "center": contour_center,
                "distance_to_center": distances_to_center,
            }
        )

    # Sort contours by closest to farthest from image centre and select the most central contour
    sorted_cells = sorted(segmented, key=lambda i: i["distance_to_center"])
    cntr_contour = sorted_cells[0]["contour"]

    return cntr_contour


def create_crop_mask_targets(img: np.ndarray,
                             ref: np.ndarray,
                             cell_img: np.ndarray,
                             pts: np.ndarray,
                             crop: bool,
                             pad: float = 50):
    """
    Create mask from shape defined by pts and crop images with defined padding.
    """
    if crop and isinstance(pts, np.ndarray):
        # Create mask to select ROI and crop images
        img, ref, cell_img, mask = _crop_roi(img, ref, cell_img, pts, pad)
        return img, ref, cell_img, mask

    if not crop and isinstance(pts, np.ndarray):
        # Create mask from points with the same size as original image
        mask = _create_mask(cell_img, pts)
        return img, ref, cell_img, mask

    else:
        mask = np.full((cell_img).shape, 255)
        return img, ref, cell_img, mask


def _crop_roi(img: np.ndarray,
              ref: np.ndarray,
              cell_img: np.ndarray,
              pts: np.ndarray,
              pad: float = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Crop images to shape defined by pts.
    """
    # Draw closed polygon defined by pts on cell image
    cv2.polylines(cell_img,[pts],True, (255), thickness=3)  # ToDo: Unexpected behaviour error

    # Calculate the coordinates and dimensions of the smallest rectangle that contains the entire polygon
    x, y, w, h = cv2.boundingRect(pts)

    # Crop original bead images to the rectangle while adding a constant padding value
    img_crop = img[y - pad:y + h + pad, x - pad:x + w + pad]
    ref_crop = ref[y - pad:y + h + pad, x - pad:x + w + pad]

    # Create mask from rectangle
    mask = _create_mask(cell_img, pts)

    # Crop mask to include padding value from before
    mask_crop = mask[y - pad:y + h + pad, x - pad:x + w + pad]

    # Crop original BF image to the rectangle while adding a constant padding value
    cell_img_crop = cell_img[y - pad:y + h + pad, x - pad:x + w + pad]

    return img_crop, ref_crop, cell_img_crop, mask_crop


# crop targets
def _create_mask(cell_img: np.ndarray, pts: np.ndarray):
    """
    Create mask from area bordered by points on black background.
    """
    return cv2.fillPoly(np.zeros(cell_img.shape), [pts], (255))  # ToDo: Unexpected behaviour error
