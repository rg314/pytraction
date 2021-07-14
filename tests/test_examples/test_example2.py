from pytraction import TractionForceConfig, process_stack

import os


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

