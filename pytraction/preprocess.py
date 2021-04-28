import cv2 
import numpy as np 
from typing import Union, Type, Tuple
from shapely import geometry
from scipy.spatial import distance

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
def _get_min_window_size(bead_img: np.ndarray, config: Type['TractionForceConfig']) -> int:
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
def _cnn_segment_cell(cell_img: np.ndarray, config: Type['TractionForceConfig']) -> np.ndarray:
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
        np.ndarray: Array of contours.
    """
    contours, _ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    return contours

def _located_most_central_cell(contours: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Calculates the distance from a contour's center of mass to the center of
    the image and returns the closed contour to the center.

    Args:
        contours (np.ndarray): Array of contours to best measured
        mask (np.ndarray): The mask to compute the central position

    Returns:
        np.ndarray: Contour closest to the center of the image.
    """

    image_center = np.asarray(mask.shape) / 2
    image_center = tuple(image_center.astype('int32'))

    segmented = []
    for contour in contours:
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

def _predict_roi(cell_img: np.ndarray, config: Type['TractionForceConfig']) -> tuple:
    """Predicts the ROI nearest the center of the cell_img. A binary mask is first
    predicted using a CNN, a counters algorithm is used to segment individual
    cell instances and the instance closest to the center of the image is used
    as the ROI.

    Args:
        cell_img (np.ndarray): Image of cell/cells to predict ROI
        config (Type[TractionForceConfig]): TractionForceConfig object

    Returns:
        tuple: (x,y) coordinates of predicted ROI
    """
    # segment image 
    mask = _cnn_segment_cell(cell_img, config)
    # get instance outlines 
    contours = _detect_cell_instances_from_segmentation(mask)
    # find most central cell
    pts = _located_most_central_cell(contours, mask)
    # get roi
    polyx, polyy = np.squeeze(pts, axis=1).T
    return polyx, polyy


def _get_polygon_and_roi(cell_img: np.ndarray, roi: tuple, 
                         config: Type['TractionForceConfig']) -> Tuple[geometry.polygon.Polygon, np.ndarray]:
    """Computes shapely polygon and pts of ROI if given. 

    Args:
        cell_img (np.ndarray): Image of cell/cells to predict ROI
        roi (tuple): ROI of current frame or None
        config (Type['TractionForceConfig']): TractionForceConfig object

    Returns:
        Tuple[geometry.polygon.Polygon, np.ndarray]: (Polygon of ROI, polygon
        pts) or None, if not ROI is provided and segmentation is not enabled in
        the config.
    """
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
def _create_mask(cell_img: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Create binary mask based on polygon pts.

    Args:
        cell_img (np.ndarray): Target image to create mask for.
        pts (np.ndarray): Points for polygon.

    Returns:
        np.ndarray: Binary mask of target polygon.
    """
    return cv2.fillPoly(np.zeros(cell_img.shape), [pts], (255))


def _crop_roi(img: np.ndarray, ref: np.ndarray, cell_img: np.ndarray, 
    pts: np.ndarray, pad=50) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    """Crops input images with ROI fitted to a bounding box. Input images are cropped to the
    polygon with a 50 pixel (default) pad.

    Args:
        img (np.ndarray): Image of beads
        ref (np.ndarray): Image of reference beads
        cell_img (np.ndarray): Image of cell
        pts (np.ndarray): Points of polygon
        pad (int, optional): Amount of pad to put around the polygon. Defaults to 50.

    Returns:
        Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]: Tuple of cropped
        images.
    """
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

def _create_crop_mask_targets(img: np.ndarray, ref: np.ndarray, cell_img: np.ndarray, 
    pts: np.ndarray, crop: bool, pad=50) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    """Crops input images with ROI fitted to a bounding box if pts are not None. Input images are cropped to the
    polygon with a 50 pixel (default) pad.

    Args:
        img (np.ndarray): Image of beads
        ref (np.ndarray): Image of reference beads
        cell_img (np.ndarray): Image of cell
        pts (np.ndarray): Points of polygon
        crop (bool): [description]
        pad (int, optional): Amount of pad to put around the polygon. Defaults to 50.

    Returns:
        Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]: Tuple of cropped
        images.
    """

    if crop and isinstance(pts, np.ndarray):
        img, ref, cell_img, mask = _crop_roi(img, ref, cell_img, pts, pad)
        return img, ref, cell_img, mask

    if not crop and isinstance(pts, np.ndarray):
        mask = _create_mask(cell_img, pts)
        return img, ref, cell_img, mask
    
    else:
        return img, ref, cell_img, None 
