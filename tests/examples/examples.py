from pytraction import TractionForceConfig, process_stack

import os
import numpy as np
from skimage import io


def test_example1():
    pix_per_mu = 1.3
    E = 100 # Young's modulus in Pa
    config = os.path.join('config', 'config.yaml')

    img_path = os.path.join('data', 'example1', 'e01_pos1_axon1.tif')
    ref_path = os.path.join('data', 'example1', 'e01_pos1_axon1_ref.tif')

    traction_config = TractionForceConfig(E=E, scaling_factor=pix_per_mu, config=config)

    img, ref, _ = traction_config.load_data(img_path, ref_path)

    dataset = process_stack(img[:1, :,:,:], ref, traction_config)

    print(dataset)

def test_example2():
    pix_per_mu = 1
    E = 3000 # Young's modulus in Pa
    config = os.path.join('config', 'config.yaml')

    img_path = os.path.join('data', 'example2', '1kPa-2-Position006.tif')
    ref_path = os.path.join('data', 'example2', '1kPa-2-Position006_ref.tif')
    roi_path = os.path.join('data', 'example2', '1kPa-2-Position006.roi')

    traction_config = TractionForceConfig(E=E, scaling_factor=pix_per_mu, config=config)

    img, ref, roi = traction_config.load_data(img_path, ref_path, roi_path)

    dataset = process_stack(img[:1, :,:,:], ref, traction_config, roi=roi, crop=True)

    print(dataset)

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

    print(dataset)



test_example1()
test_example2()
test_example3()