import glob
import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from skimage import io

from pytraction.utils import allign_slice
from pytraction.piv import PIV
from pytraction.fourier import fourier_xu

from openpiv import tools, pyprocess, validation, filters, scaling 


frame = 0
channel = 0

meshsize = 10 # grid spacing in pix
pix_durch_mu = 1.3

E = 100 # Young's modulus in Pa
s = 0.3 # Poisson's ratio


# load data from ryan
rg_file = glob.glob('data/*')


img1 = 'data/e01_pos1_axon1.tif'
ref1 = 'data/e01_pos1_axon1_ref.tif'

img1 = io.imread(img1)
ref1 = io.imread(ref1)

frame_a= np.array(img1[frame, channel, :, :], dtype='uint8')
frame_b = np.array(ref1[channel,:,:], dtype='uint8')

frame_a = allign_slice(frame_a, frame_b)

# img = np.stack([frame_b, frame_a])
# io.imsave('test.tif',img)

piv_obj = PIV(window_size=32, search_area_size=32, overlap=16)
x, y, u, v = piv_obj.base_piv(frame_a, frame_b)



plt.quiver(x,y,u,v)
plt.show()
