from pytraction.utils import (
    align_slice,
    sparse_cholesky,
    interp_vec2grid, 
    normalize,
    clahe,
    bead_density,
    plot,
    )

import numpy as np
from scipy.ndimage.filters import gaussian_filter
import cv2 

def test_align_slice():
    
    # generate 300 particles
    x = np.random.randint(10, 246, 300)
    y = np.random.randint(10, 246, 300)
    
    # get x and y sizes
    xsize = int(np.max(x))
    ysize = int(np.max(y))
    
    # create image with no particles
    img = np.zeros((xsize, ysize))

    # for each particle at [i,j] make a square
    for i, j in zip(x,y):
        # get random bead radius with center i,j
        particle_radius = int(np.random.randint(3, 6, 1))
        cx, cy = int(i)-1, int(j)-1
        # make particle at i,j with radius r
        img[cx-particle_radius: cx+particle_radius, cy-particle_radius: cy+particle_radius] = 255

    # apply gaussian blur to filter
    img = gaussian_filter(img, sigma=3)
    # generate random noise
    noise = np.random.randint(0, 125, size=xsize*ysize).reshape(((xsize, ysize)))
    # combine noise + img
    img = img + noise
    # make sure the image is between [0-255] and uint8
    img = np.array(img*255/np.max(img), dtype='uint8')

    # boarder to apply to input image
    pad = 30

    # possible translational drifts
    drifts = {
        'top-right':[5,10], 
        'top-left':[-5,10], 
        'bottom-left':[-5,-10], 
        'bottom-right':[5,-10]
        }

    # check different drifts are corrected
    for k, (dx0, dy0) in drifts.items():
        ref_target = img[pad:-pad, pad:-pad]
        img_target = img[pad+dy0:-pad+dy0, pad+dx0:-pad+dx0]
        # note that dx and dy are a measure of how much the target image has drifted from the refernece
        dx, dy, aligned_img = align_slice(img_target, ref_target)

        assert dx0 == dx, f'{k} shift did not align correctly'
        assert dy0 == dy, f'{k} shift did not align correctly'

def test_normalize():
    targets = {
        'img_float':np.random.rand(512*512).reshape((512,512)),
        'img_below_255':np.random.randint(0, 255, 512*512).reshape((512,512)),
        'img_255':np.random.randint(0, 127, 512*512).reshape((512,512)),
        'img_above_255':np.random.randint(0, 20000, 512*512).reshape((512,512)),
    }

    for name, img_target in targets.items():

        img = normalize(img_target)

        assert img.dtype == np.uint8, 'normalize did not return the correct dtype'
        assert np.min(img) == 0, 'normalize did not scale down to 0'
        assert np.max(img) == 255, 'normalize did not scale up to 255'

