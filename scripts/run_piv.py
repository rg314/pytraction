import glob
import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

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

E = 100 # Young's modulus in Pa
s = 0.3 # Poisson's ratio


df = pd.read_csv('data\PIV_test.txt', delimiter=' ', header=None)
df = df.iloc[:,:-1]

df.columns = ['x', 'y', 'ux1', 'uy1', 'mag1', 'ang1', 'p1', 'ux2', 'uy2', 'mag2', 'ang2', 'p2', 'ux0', 'uy0', 'mag0', 'flag']
x, y, u, v = df[['x', 'y', 'ux1', 'uy1']].T.values

rgcopy = [x, y, u, v]



noise = 20
xn, yn, un, vn = x[:noise],y[:noise],u[:noise],v[:noise]
noise_vec = np.array([un.flatten(), vn.flatten()])

# fig, ax = plt.subplots(1,2)

# ax[0].quiver(x,y,u,v)
# ax[1].quiver(xn,yn,un,vn)
# plt.show()


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

print(L)


img = traction_magnitude.reshape(i_max, j_max).T


ax = plt.subplot(111)
x, y, u, v = rgcopy
img = np.flip(img, axis=0)
im = ax.imshow(img, interpolation='bicubic', cmap='jet',extent=[x.min(), x.max(), y.min(), y.max()] )
ax.quiver(x, y, u, v)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)
plt.show()
