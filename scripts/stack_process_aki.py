import glob
import os
import cv2
import numpy as np
import pandas as pd
from skimage import io
import matplotlib.pyplot as plt
from shapely import geometry


from pytraction.piv import PIV
from pytraction.utils import normalize
from pytraction.traction_force import PyTraction
from read_roi import read_roi_file
import pytraction.net.segment as pynet 

from mpl_toolkits.axes_grid1 import make_axes_locatable

from scripts.utils import get_model, read_aki_data_stacks

channel = 0
scaling_factor = 9.8138
TFM = True
segment = True
device = 'cpu'
roi = None
WINDOW_SIZE = 16
fmax_plot = 175


# load data from ryan
path = 'data/Blebbistatin HelaKyoto/'
files = glob.glob(f'{path}**')
files = [os.path.basename(x) for x in files if '-res' not in x]


e_loopup = {
    '10kPa': 10000,
    '1kPa': 1000,
    '3kPa': 3000,
}


if segment:
    model, pre_fn = get_model(device)

for img_stack_name in files:
    # load image stacks
    img1, ref1 = read_aki_data_stacks(img_stack_name, path)
    img_stack_name = path + img_stack_name

    if os.path.exists(img_stack_name.replace('.tif', '.roi')):
        if not os.path.isdir(img_stack_name.replace('.tif', '.roi')):
            roi = read_roi_file(img_stack_name.replace('.tif', '.roi'))


    # get number of frames
    nframes = img1.shape[0]

    # save paths
    img_path, name = os.path.split(img_stack_name)
    save_path = os.path.join(img_path, name.split('.')[0]+'-res')
    e_target = name.split('-')[0]
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # set up traction object
    traction = PyTraction(
        meshsize = 10, # grid spacing in pix
        pix_per_mu = scaling_factor,
        E = e_loopup[e_target], # Young's modulus in Pa
        s = 0.5, # Poisson's ratio
    )

    print(f'Target stiffness lookup of {e_target}. Init traction object with stiffness {traction.E}')

    for frame in range(nframes):
        # load plane
        ref = normalize(np.array(ref1[channel,:,:]))
        img = normalize(np.array(img1[frame, channel, :, :]))

        if segment:
            cell_img = np.array(img1[frame, 1, :, :])
            cell_img = normalize(cell_img)
            mask = pynet.get_mask(cell_img, model, pre_fn, device=device)

            mask = np.array(~mask.astype('bool'), dtype='uint8')

            contours, _ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            areas = [cv2.contourArea(c) for c in contours]
            sorted_areas = np.sort(areas)

            #bounding box (red)
            cnt=contours[areas.index(sorted_areas[-1])] #the biggest contour

            cv2.drawContours(cell_img, [cnt], -1, (255), 1, cv2.LINE_AA)

            polyx, polyy = np.squeeze(cnt, axis=1).T
            roi = True


        # if there's a ROI file provided cut the cell out and pad with 0.2x cell max len and crop input images 
        if roi: 
            shift=0.2
            if not segment:
                polyx = roi[name.split('.')[0]]['x']
                polyy = roi[name.split('.')[0]]['y']

            minx, maxx = np.min(polyx), np.max(polyx)
            miny, maxy = np.min(polyy), np.max(polyy)

            midx = minx + (maxx-minx) // 2
            midy = miny + (maxy-miny) // 2

            pixel_shift = int(max(midx, midy) * shift) // 2

            rescaled = []
            for (xi,yi) in zip(polyx, polyy):
                if xi < midx:
                    x_shift = xi - pixel_shift
                else:
                    x_shift = xi + pixel_shift

                if yi < midy:
                    y_shift = yi - pixel_shift
                else:
                    y_shift = yi + pixel_shift
                
                rescaled.append([x_shift, y_shift])


            polygon = geometry.Polygon(rescaled)

            x,y,w,h = cv2.boundingRect(np.array(rescaled))
            pad = 50

            img = img[y-pad:y+h+pad, x-pad:x+w+pad]
            ref = ref[y-pad:y+h+pad, x-pad:x+w+pad]

            if not segment:
                cell_img = np.array(img1[frame, 1, :, :])
                pts = np.array(list(zip(polyx, polyy)), np.int32)
                pts = pts.reshape((-1,1,2))
                cv2.polylines(cell_img,[pts],True,(255), thickness=3)
            
            cell_img_full = cell_img
            cell_img = cell_img[y-pad:y+h+pad, x-pad:x+w+pad]



        # do piv
        try:
            piv_obj = PIV(window_size=WINDOW_SIZE)
            tmp = f'{save_path}/corr_{frame}.tif'
            x, y, u, v, stack = piv_obj.iterative_piv(img, ref)

        except:
            continue
        

        
        if not TFM:
            mag = np.sqrt(u**2, v**2)
            fig, ax = plt.subplots(1,2)
            ax[0].quiver(x,y,u,v)
            ax[1].hist(mag.flatten())
            plt.show()


        if TFM:
            if frame == 0 and not roi:
#                 # get beta to compute baysian prior
                noise = 10
                xn, yn, un, vn = x[:noise],y[:noise],u[:noise],v[:noise]
                noise_vec = np.array([un.flatten(), vn.flatten()])

                varnoise = np.var(noise_vec)
                beta = 1/varnoise
            
            elif roi:
                noise = []
                for (x0,y0, u0, v0) in zip(x.flatten(),y.flatten(), u.flatten(), v.flatten()):
                    p1 = geometry.Point([x0,y0])
                    if not p1.within(polygon):
                        noise.append(np.array([u0, v0]))

                noise_vec = np.array(noise)
                varnoise = np.var(noise_vec)
                beta = 1/varnoise

            # make pos and vecs for TFM
            pos = np.array([x.flatten(), y.flatten()])
            vec = np.array([u.flatten(), v.flatten()])

            try:

                # compute traction map
                traction_map, f_n_m, L = traction.calculate_traction_map(pos, vec, beta)


                # plot traction map
                fig, ax = plt.subplots(1,3)
                im1 = ax[0].imshow(traction_map, interpolation='bicubic', cmap='jet',extent=[x.min(), x.max(), y.min(), y.max()], vmin=0, vmax=fmax_plot)
                ax[0].quiver(x, y, u, v)
                divider1 = make_axes_locatable(ax[0])
                cax1 = divider1.append_axes("right", size="5%", pad=0.05)

                im2 = ax[1].imshow(cell_img, cmap='gray',vmax=np.max(cell_img))
                im3 = ax[2].imshow(cell_img_full, cmap='gray',vmax=np.max(cell_img))
                
                
                fig.colorbar(im1, cax=cax1)

                ax[0].set_axis_off()
                ax[1].set_axis_off()
                ax[2].set_axis_off()
                plt.suptitle(f'{name} for frame {frame}')
                plt.tight_layout()
                plt.savefig(f'{save_path}/{frame}.png', dpi=300)
                plt.close()
            
            except Exception as e:
                print(e)



# print('Completed')
