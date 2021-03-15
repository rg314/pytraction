import glob
import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os 
from skimage import io

from pytraction.utils import allign_slice
from pytraction.piv import PIV
from pytraction.fourier import fourier_xu
from pytraction.reg_fourier import reg_fourier_tfm
from pytraction.optimal_lambda import optimal_lambda

from openpiv import tools, pyprocess, validation, filters, scaling 
from scipy.io import loadmat
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

frame = 0
channel = 0

meshsize = 10 # grid spacing in pix
pix_durch_mu = 1.3

E = 10000 # Young's modulus in Pa
s = 0.3 # Poisson's ratio

df = pd.read_csv(f'data{os.sep}matlab_data.csv')

un, vn, x, y, u, v = df.T.values



noise_vec = np.array([un.flatten(), vn.flatten()])



varnoise = np.var(noise_vec)
beta = 1/varnoise



pos = np.array([x.flatten(), y.flatten()])
vec = np.array([u.flatten(), v.flatten()])


# calculate with E=1 and rescale Young's modulus at the end
# prepare Fourier-tranformed Green's function and displacement
# NOTE: fourier_X_u shifts the positions of the displacement
# vectors such that the tractions are displayed in the deformed frame
# where the cell is localized
grid_mat, i_max, j_max, X, fuu, Ftux, Ftuy, u = fourier_xu(pos,vec, meshsize, 1, s,[])

# get lambda from baysian bad boi 
L, evidencep, evidence_one = optimal_lambda(beta, fuu, Ftux, Ftuy, 1, s, meshsize, i_max, j_max, X, 1)

# do the TFM
pos,traction,traction_magnitude,f_n_m,_,_ = reg_fourier_tfm(Ftux, Ftuy, L, 1, s, meshsize, i_max, j_max, grid_mat, pix_durch_mu, 0)


#rescale traction with proper Young's modulus
traction = E*traction
traction_magnitude = E*traction_magnitude
f_n_m = E*f_n_m

#display a heatmap of the spatial traction distribution
# fnorm = np.sqrt(f_n_m[:,:,1]**2 + f_n_m[:,:,0]**2)**0.5

print('the parameter was', L)

img = traction_magnitude.reshape(i_max, j_max).T

ax = plt.subplot(111)
im = ax.imshow(img, interpolation='bicubic', cmap='jet')

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)
plt.show()
