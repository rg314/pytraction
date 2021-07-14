from pytraction import TractionForceConfig, process_stack

import os


def test_example1():
    pix_per_mu = 1.3
    E = 100 # Young's modulus in Pa
    config = os.path.join('config', 'config.yaml')

    img_path = os.path.join('data', 'example1', 'e01_pos1_axon1.tif')
    ref_path = os.path.join('data', 'example1', 'e01_pos1_axon1_ref.tif')

    traction_config = TractionForceConfig(E=E, scaling_factor=pix_per_mu, config=config)

    img, ref, _ = traction_config.load_data(img_path, ref_path)

    dataset = process_stack(img[:1, :,:,:], ref, traction_config)

