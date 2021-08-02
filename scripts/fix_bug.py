from matplotlib import axes
from pytraction import TractionForceConfig
from pytraction import plot, process_stack
import matplotlib.pyplot as plt
import glob
from skimage import io
from tifffile import tifffile
import numpy as np
import shutil
import cv2

pix_per_mu = 1.3 # The number of pixels per micron 
E = 100 # Youngs modulus in Pa
config = 'config/config.yaml'
pos = 'pos_0'
crop_status = 'old'
run = False


if run:
    for pos in ['pos_0', 'pos_3', 'pos_12']:


        files = glob.glob('data/lew/*')
        files = [x for x in files if pos in x]


        cell_img = [x for x in files if 'ch00_cr' in x][0]
        bead_img = [x for x in files if 'wc_ch01' in x][0]
        ref_img = [x for x in files if 'woc_ch01' in x][0]

        cell_img = io.imread(cell_img)
        bead_img = io.imread(bead_img)
        ref_img = io.imread(ref_img)

        img = np.stack([[bead_img, cell_img]])
        img_path = 'data/lew/proc/'+ pos+'_img.tif'
        tifffile.imsave(img_path, img)

        ref = np.stack([ref_img, ref_img])
        ref_path = 'data/lew/proc/'+ pos+'_ref.tif'
        tifffile.imsave(ref_path, ref)

        roi = [x for x in files if 'roi' in x][0]
        roi_path = 'data/lew/proc/'+ pos+'.roi'
        shutil.copyfile(roi, roi_path)


        traction_config = TractionForceConfig(E=E, scaling_factor=pix_per_mu,config=config)
        img, ref, _ = traction_config.load_data(img_path, ref_path, roi_path=roi_path)
        log = process_stack(img, ref, config=traction_config)
        log.save('data/lew/proc/'+ pos+crop_status+'.h5')


        plot(log, frame=0)
        plt.savefig('data/lew/proc/'+ pos + crop_status+'.png')
        plt.close()


files = glob.glob('data/lew/proc/*.png')
files = [x for x in files if 'old' in x]

fig , ax = plt.subplots(2,3)

for idx, old in enumerate(files):
    new = old.replace('old', 'new')

    new_img = cv2.imread(new)
    old_img = cv2.imread(old)

    
    ax[0,idx].imshow(old_img)
    ax[0,idx].set_title('Old')
    ax[1,idx].imshow(new_img)
    ax[1,idx].set_title('New')

[ax_i.set_axis_off() for ax_i in ax.ravel()]
plt.show()
