import cv2 
import numpy as np 
from typing import Union, Type, Tuple
from shapely import geometry
from scipy.spatial import distance

from pytraction import TractionForceConfig
from pytraction.net import segment as pynet
from pytraction.utils import normalize, allign_slice, bead_density


# get the images of interest
def _get_reference_frame(ref_stack: np.ndarray, bead_channel: int) -> np.ndarray:
    """Normalizes the reference frame and drops all other axes except the bead
    channel and returns a (w, h) np.ndarray.

    Args:
        ref_stack (numpy.ndarray): Reference image stack with shape (c, w, h).
        bead_channel (int): Channel, c of reference bead image, bead_channel < c. 

    Returns:
        numpy.ndarray: Reference bead frame with shape (w,h) dtype uint8.
    """
    
    assert bead_channel < ref_stack.shape[0], ValueError(f'bead_channel > number of image channels')

    return normalize(np.array(ref_stack[bead_channel,:,:]))


def _get_img_frame(img_stack: np.ndarray, frame:int, bead_channel:int) -> np.ndarray:
    """Normalizes the cell image (frame) and drops all other axes except the
    cell image channel and returns a (w, h) np.ndarray.

    Args:
        img_stack (numpy.ndarray): Image stack with shape (t, c, w, h).
        frame (int): Image stack of at time frame.
        bead_channel (int): Channel, c of frame bead image, bead_channel < c. 

    Returns:
        numpy.ndarray: Image of cell frame at time (frame) with shape (w,h) dtype uint8 at frame.
    """
    assert bead_channel < img_stack.shape[0], ValueError(f'frame > number of time stacks')
    assert bead_channel < img_stack.shape[1], ValueError(f'bead_channel > number of image channels')
    
    return normalize(np.array(img_stack[frame, bead_channel, :, :]))

def _get_cell_img(img_stack: np.ndarray, frame: int, cell_channel: int) -> np.ndarray: 
    """Normalizes the cell image (frame) and drops all other axes except the
    cell image channel and returns a (w, h) np.ndarray.

    Args:
        img_stack (numpy.ndarray): Image stack with shape (t, c, w, h).
        frame (int): Image stack of at time frame.
        cell_channel (int): Channel, c of cell image, cell_channel < c. 

    Returns:
        numpy.ndarray: Image of cell frame at time (frame) with shape (w,h) dtype uint8 at frame.
    """
    assert cell_channel < img_stack.shape[0], ValueError(f'frame > number of time stacks')
    assert cell_channel < img_stack.shape[1], ValueError(f'cell_channel > number of image channels')
    
    return normalize(np.array(img_stack[frame, cell_channel, :, :]))


def _get_raw_frames(img_stack: np.ndarray, ref_stack: np.ndarray, 
    frame: np.ndarray, bead_channel: int, cell_channel: int) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    """Loads in image data for processing including bead and reference frame and
    image of cell. 

    Args:
        img_stack (np.ndarray): Bead image stack to process (t,c,w,h)
        ref_stack (np.ndarray): Bead reference image stack to process (c,w,h)
        frame (np.ndarray): Current frame to process
        bead_channel (int): Channel of beads (c)
        cell_channel (int): Channel of cell (c)

    Returns:
        Tuple[np.ndarray,np.ndarray,np.ndarray]: Preprocessed (img, ref,
        cell_img).
    """

    img = _get_img_frame(img_stack, frame, bead_channel)
    ref = _get_reference_frame(ref_stack, bead_channel)
    cell_img = _get_cell_img(img_stack, frame, cell_channel)
    return img, ref, cell_img


# get the window size
def _get_min_window_size(bead_img: np.ndarray, config: Type[TractionForceConfig]) -> int:
    """Computes the bead density and uses KNN model to predict min_window_size
    for PIV analysis.

    Args:
        bead_img (np.ndarray): Target bead image to measure displacements
        config (Type[TractionForceConfig]): TractionForceConfig object

    Returns:
        int: Minimum window size for PIV.
    """
    if not config.config['piv']['min_window_size']:
        density = bead_density(bead_img)

        knn = config.knn
        min_window_size = knn.predict([[density]])
        print(f'Automatically selected window size of {min_window_size}')

        return int(min_window_size)
    else:
        return config.config['piv']['min_window_size']


# load frame roi
def _load_frame_roi(roi: Union[list, tuple], frame: int, nframes: int) -> tuple:
    """Loads ROIs for frame from a list of ROIs or tuple.

    Args:
        roi (Union[list, tuple]): List of ROIs [(x0...,y0...), (x1...,y1...)] or
        tuple of ROI (x0..., y0...)
        frame (int): Current frame
        nframes (int): Total number of frame in a stack

    Returns:
        tuple: A single ROI to process (x0..., y0...)
    """

    if isinstance(roi, list):
        assert len(roi) == nframes, f'Warning ROI list has len {len(roi)} which is not equal to \
                                    the number of frames ({nframes}). This would suggest that you do not \
                                    have the correct number of ROIs in the zip file.'
        return roi[frame]
    else:
        return roi

# get roi
def _cnn_segment_cell(cell_img: np.ndarray, config: Type[TractionForceConfig]) -> np.ndarray:
    """Segment cell / cells from grayscale image using CNN.

    Args:
        cell_img (np.ndarray): Input grayscale image of cell frame
        config (Type[TractionForceConfig]): TractionForceConfig object

    Returns:
        np.ndarray: Binary mask of segmented cell.
    """

    mask = pynet.get_mask(cell_img, config.model, config.pre_fn, device=config.device)
    return np.array(mask.astype('bool'), dtype='uint8')


def _detect_cell_instances_from_segmentation(mask: np.ndarray) -> np.ndarray:
    """Find contours from a binary mask.

    Args:
        mask (np.ndarray): Binary mask of segmented cells.

    Returns:
        np.ndarray: Array of counters.
    """
    contours, _ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    return contours

def _located_most_central_cell(counters: np.ndarray, mask: np.ndarray) -> np.ndarray:

    image_center = np.asarray(mask.shape) / 2
    image_center = tuple(image_center.astype('int32'))

    segmented = []
    for contour in counters:
        # find center of each contour
        M = cv2.moments(contour)
        center_X = int(M["m10"] / M["m00"])
        center_Y = int(M["m01"] / M["m00"])
        contour_center = (center_X, center_Y)
    
        # calculate distance to image_center
        distances_to_center = (distance.euclidean(image_center, contour_center))
    
        # save to a list of dictionaries
        segmented.append({
            'contour': contour, 
            'center': contour_center, 
            'distance_to_center': distances_to_center
            }
            )

    sorted_cells = sorted(segmented, key=lambda i: i['distance_to_center'])
    pts=sorted_cells[0]['contour'] #the biggest contour
    return pts

def _predict_roi(cell_img, config):
    # segment image 
    mask = _cnn_segment_cell(cell_img, config)
    # get instance outlines 
    contours = _detect_cell_instances_from_segmentation(mask)
    # find most central cell
    pts = _located_most_central_cell(contours, mask)
    # get roi
    polyx, polyy = np.squeeze(pts, axis=1).T
    return polyx, polyy


def _get_polygon_and_roi(cell_img, roi, config):
    if config.config['settings']['segment']:
        polyx, polyy = _predict_roi(cell_img, config)
        pts = np.array(list(zip(polyx, polyy)), np.int32)
        polygon = geometry.Polygon(pts)
        pts = pts.reshape((-1,1,2))
        return polygon, pts


    elif roi:
        polyx, polyy = roi[0], roi[1]
        pts = np.array(list(zip(polyx, polyy)), np.int32)
        polygon = geometry.Polygon(pts)
        pts = pts.reshape((-1,1,2))
        return polygon, pts
    
    else:
        return None, None



# crop targets
def _create_mask(cell_img, pts):
    return cv2.fillPoly(np.zeros(cell_img.shape), [pts], (255))


def _crop_roi(img, ref, cell_img, pts, pad=50):
    # draw polygon on cell image
    cv2.polylines(cell_img,[pts],True,(255), thickness=3)

    # get bounding box for polygon
    x,y,w,h = cv2.boundingRect(pts)
    
    # crop img/ref
    img_crop = img[y-pad:y+h+pad, x-pad:x+w+pad]
    ref_crop = ref[y-pad:y+h+pad, x-pad:x+w+pad]

    # create mask 
    mask = _create_mask(cell_img, pts)
    mask_crop = mask[y-pad:y+h+pad, x-pad:x+w+pad]

    # crop cell image
    cell_img_full = cell_img
    cell_img_crop = cell_img[y-pad:y+h+pad, x-pad:x+w+pad]

    return img_crop, ref_crop, cell_img_crop, mask_crop

def _create_crop_mask_targets(img, ref, cell_img, pts, crop, pad=50):
    if crop and isinstance(pts, np.ndarray):
        img, ref, cell_img, mask = _crop_roi(img, ref, cell_img, pts, pad)
        return img, ref, cell_img, mask

    if not crop and isinstance(pts, np.ndarray):
        mask = _create_mask(cell_img, pts)
        return img, ref, cell_img, mask
    
    else:
        return img, ref, cell_img, None 
