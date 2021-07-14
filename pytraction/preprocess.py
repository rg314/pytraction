import numpy as np 
import cv2 
from shapely import geometry
from scipy.spatial import distance

from pytraction.net import segment as pynet
from pytraction.utils import normalize, allign_slice, bead_density


# get the images of interest
def _get_reference_frame(ref_stack, _, bead_channel):
    return normalize(np.array(ref_stack[bead_channel,:,:]))

def _get_img_frame(img_stack, frame, bead_channel):
    return normalize(np.array(img_stack[frame, bead_channel, :, :]))

def _get_cell_img(img_stack, frame, cell_channel):
    return normalize(np.array(img_stack[frame, cell_channel, :, :]))

def _get_raw_frames(img_stack, ref_stack, frame, bead_channel, cell_channel):
    img = _get_img_frame(img_stack, frame, bead_channel)
    ref = _get_reference_frame(ref_stack, frame, bead_channel)
    cell_img = _get_cell_img(img_stack, frame, cell_channel)
    return img, ref, cell_img


# get the window size
def _get_min_window_size(img, config):
    if not config.config['piv']['min_window_size']:
        density = bead_density(img)

        knn = config.knn
        min_window_size = knn.predict([[density]])
        print(f'Automatically selected window size of {min_window_size}')

        return int(min_window_size)
    else:
        return config.config['piv']['min_window_size']


# load frame roi
def _load_frame_roi(roi, frame, nframes):
    if isinstance(roi, list):
        assert len(roi) == nframes, f'Warning ROI list has len {len(roi)} which is not equal to \
                                    the number of frames ({nframes}). This would suggest that you do not \
                                    have the correct number of ROIs in the zip file.'
        return roi[frame]
    else:
        return roi

# get roi
def _cnn_segment_cell(cell_img, config):
    mask = pynet.get_mask(cell_img, config.model, config.pre_fn, device=config.config['settings']['device'])
    return np.array(mask.astype('bool'), dtype='uint8')


def _detect_cell_instances_from_segmentation(mask):
    contours, _ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    return contours

def _located_most_central_cell(counters, mask):
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
