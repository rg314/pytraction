import glob
import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from pytraction.utils import allign_slice

from openpiv import tools, pyprocess, validation, filters, scaling 

# get sample data from Aki 

# point and search the correct path
sep = os.sep
files = glob.glob(f'building{sep}Aki data{sep}PositionNTC310kPa19{sep}*')

# get img, cell outline, and piv files 
image = [x for x in files if '.tif' in x][0]
outline = [x for x in files if '.csv' in x][0]
piv_aki = [x for x in files if '.txt' in x][0]

# read in the files and format to known format / shape
outline = pd.read_csv(outline).values # reads in x and y as [x,y]
outline = outline.reshape((-1,1,2)).astype(np.int32)
img = cv2.imread(image, 0)
piv_aki = pd.read_csv(piv_aki, delimiter='\s', header=None, engine='python')


# load data from ryan
rg_file = glob.glob('data/*')

img1 = [x for x in rg_file if 'ref' not in x][0]
ref1 = [x for x in rg_file if 'ref' in x][0]

img1 = io.imread(img1)
ref1 = io.imread(ref1)

frame = 0
channel = 0

frame_a= np.array(img1[frame, channel, :, :], dtype='uint8')
frame_b = np.array(ref1[channel,:,:], dtype='uint8')


print(frame_a.shape)
frame_a = allign_slice(frame_a, frame_b)
print(frame_a.shape)

img = np.stack([frame_b, frame_a])

# io.imsave('test.tif',img)

def base_piv(img, ref, scaling_factor = 1, window_size=64, search_area_size=64, overlap=32, dt=1):
    u, v, sig2noise = pyprocess.extended_search_area_piv( 
        pyprocess.normalize_intensity(ref), 
        pyprocess.normalize_intensity(img), 
        window_size=window_size, 
        overlap=overlap, 
        dt=dt, 
        search_area_size=search_area_size, 
        sig2noise_method='peak2peak')

    # prepare centers of the IWs to know where locate the vectors
    x, y = pyprocess.get_coordinates(img.shape, 
                                    search_area_size=search_area_size, 
                                    overlap=overlap)

    u, v, mask = validation.sig2noise_val( u, v, 
                                        sig2noise, 
                                        threshold = np.percentile(sig2noise,5))

    # removing and filling in the outlier vectors
    u, v = filters.replace_outliers( u, v, method='localmean', 
                                    max_iter=10, 
                                    kernel_size=2)

    # rescale the results to millimeters and mm/sec
    x, y, u, v = scaling.uniform(x, y, u, v, 
                                scaling_factor=scaling_factor )

    # save the data
    x, y, u, v = tools.transform_coordinates(x, y, u, v)

    return x, y, u, v
    

for x in [256, 128, 64, 32, 16]:
    x, y, u, v = base_piv(frame_a, frame_b, window_size=x, search_area_size=x, overlap=x//2)
    m = np.sqrt(u**2 + v**2).flatten()
    fig, ax = plt.subplots(1,2)
    ax[0].hist(m)
    ax[1].imshow(frame_a, cmap='gray')
    ax[1].quiver(x,y, u,v)
    plt.show()