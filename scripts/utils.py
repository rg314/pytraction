import glob
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import os 

import torch
import segmentation_models_pytorch as smp
from pytraction.net.dataloader import get_preprocessing

def read_aki_data_stacks(folder, data_path):
    files = glob.glob(f'{data_path}{folder}/*.tif')

    assert files

    files = sorted(files)

    def z_project(img):
        img = io.imread(img)
        img_max= np.max(img, axis=0)
        return img_max

    tzxy_stack = []
    for file in files:
        if 'start' not in file.lower() and 'stop' not in file.lower() and 'beads' in file.lower():
            beads = file
            cell = file.replace('Beads', 'Cell')
            
            assert os.path.exists(beads)
            assert os.path.exists(cell)
            

            cell_z = io.imread(cell)[0,:,:]
            beads_z = z_project(beads)

            stack = np.stack([beads_z, cell_z])

            tzxy_stack.append(stack)
        
        if 'stop' in file.lower() and 'beads' in file.lower():
            beads = file
            cell = file.replace('Beads', 'Cell')
            
            assert os.path.exists(beads)
            assert os.path.exists(cell)
            
            print(cell, beads)

            cell_z = z_project(cell)
            beads_z = z_project(beads)

            ref = np.stack([beads_z, cell_z])

    img = np.stack(tzxy_stack)
    return img, ref 



def get_model(device):
    # currently using model from 20210316
    best_model = torch.hub.load_state_dict_from_url(
        'https://docs.google.com/uc?export=download&id=1zShYcG8IMsMjB8hA6FcBTIZPfi_wDL4n')
    if device == 'cpu':
        best_model = best_model.to('cpu')
    else:
        best_model = best_model.to('cuda')
    preproc_fn = smp.encoders.get_preprocessing_fn('efficientnet-b1', 'imagenet')
    preprocessing_fn = get_preprocessing(preproc_fn)

    return best_model, preprocessing_fn
    