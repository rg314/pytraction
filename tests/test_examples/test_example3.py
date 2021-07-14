from pytraction import TractionForceConfig, process_stack

import os
import numpy as np


def test_example3():

    pix_per_mu = 1.3
    E = 3000
    config = os.path.join('config', 'config.yaml')

    img_path_bead = os.path.join('data', 'example3', 'Beads3.tif')
    img_path_cell = os.path.join('data', 'example3', 'Cell3.tif')
    ref_path = os.path.join('data', 'example3', 'BeadsStop.tif')
    
    def z_project(img_path):
        img = io.imread(img_path)
        img_max= np.max(img, axis=0)
        return img_max

    bead = z_project(img_path_bead)
    cell = z_project(img_path_cell)
    ref = z_project(ref_path)

    img = np.stack([[bead, cell]])
    ref = np.stack([ref, ref])

    traction_config = TractionForceConfig(E=E, scaling_factor=pix_per_mu, config=config)
    traction_config.config['settings']['segment'] = True

    dataset = process_stack(img, ref, traction_config, roi=None, crop=True)

