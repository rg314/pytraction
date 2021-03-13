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
from scipy.io import loadmat


frame = 0
channel = 0

meshsize = 10 # grid spacing in pix
pix_durch_mu = 1.3

E = 100 # Young's modulus in Pa
s = 0.3 # Poisson's ratio


data = loadmat('/Users/ryan/Documents/GitHub/Easy-to-use_TFM_package/test_data/input_data.mat')

pos, vec = data['input_data'][0][0][0][0][0]

x, y = pos.T
u, v = vec.T

# plt.quiver(x,y,u,v)
# plt.imshow(frame_a)
# plt.show()

noise = 7
# xn, yx, un, vn = x[:noise,:noise],y[:noise,:noise],u[:noise,:noise],v[:noise,:noise]
xn, yx, un, vn = x[:noise],y[:noise],u[:noise],v[:noise]
noise_vec = np.array([un.flatten(), vn.flatten()])

varnoise = np.var(noise_vec, axis=1)
beta = 1/varnoise

pos = np.array([x.flatten(), y.flatten()])
vec = np.array([u.flatten(), v.flatten()])


# calculate with E=1 and rescale Young's modulus at the end
# prepare Fourier-tranformed Green's function and displacement
# NOTE: fourier_X_u shifts the positions of the displacement
# vectors such that the tractions are displayed in the deformed frame
# where the cell is localized

grid_mat, i_max, j_max, X, fuu, Ftux, Ftuy, u = fourier_xu(pos,vec, meshsize, 1, s,[])



            

# print(x.shape)

# plt.quiver()
# plt.show()
