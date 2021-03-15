import glob
import os
import cv2
import numpy as np
import pandas as pd
from skimage import io
import matplotlib.pyplot as plt


from pytraction.piv import PIV
from pytraction.utils import allign_slice
from pytraction.traction_force import PyTraction

from mpl_toolkits.axes_grid1 import make_axes_locatable


scaling_factor = 1.3
TFM = True


# load data from ryan
files = glob.glob('data/*.tif')
images = [x for x in files if '_ref' not in x]



traction = PyTraction(
    meshsize = 10, # grid spacing in pix
    pix_per_mu = scaling_factor,
    E = 300, # Young's modulus in Pa
    s = 0.5, # Poisson's ratio
)


for img_stack_name in images:
    # load image stacks
    img1 = io.imread(img_stack_name)
    ref1 = io.imread(img_stack_name.replace('.tif', '_ref.tif'))

    # get number of frames
    nframes = img1.shape[0]

    # save paths
    img_path, name = os.path.split(img_stack_name)
    save_path = os.path.join(img_path, name.split('.')[0])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    

    for frame in range(nframes):
        # load plane
        ref = np.array(ref1[channel,:,:], dtype='uint8')
        img = np.array(img1[frame, channel, :, :], dtype='uint8')

        # do piv
        piv_obj = PIV(window_size=32, search_area_size=32, overlap=16)
        x, y, u, v, stack = piv_obj.iterative_piv(img, ref)


        if not TFM:
            mag = np.sqrt(u**2, v**2)
            fig, ax = plt.subplots(1,2)
            ax[0].quiver(x,y,u,v)
            ax[1].hist(mag.flatten())
            plt.show()


        if TFM:
            if frame == 0:
                # get beta to compute baysian prior
                noise = 10
                xn, yn, un, vn = x[:noise],y[:noise],u[:noise],v[:noise]
                noise_vec = np.array([un.flatten(), vn.flatten()])

                varnoise = np.var(noise_vec)
                beta = 1/varnoise

            # make pos and vecs for TFM
            pos = np.array([x.flatten(), y.flatten()])
            vec = np.array([u.flatten(), v.flatten()])

            # compute traction map
            traction_map, f_n_m, L = traction.calculate_traction_map(pos, vec, beta)

            # plot traction map
            ax = plt.subplot(111)
            im = ax.imshow(traction_map, interpolation='bicubic', cmap='jet',extent=[x.min(), x.max(), y.min(), y.max()], vmin=0)
            ax.quiver(x, y, u, v)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            plt.savefig(f'{save_path}/{frame}.png')
            plt.close()


print('Completed')
