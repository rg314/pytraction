import os
import numpy as np
import pandas as pd
import h5py
import io

from pytraction.core import TractionForceConfig
from pytraction.tractionforcedataset import TractionForceDataset, write_tfm_results, write_tfm_metadata


def test_tfm_results():
    elastic_modulus = 100
    scaling_factor = 1.3
    poisson_ratio = 0.48
    window_size = 16
    meshsize = 8
    config_path = os.path.join("example_data", "example_config.yaml")

    # Create config instance
    config = TractionForceConfig(
        config_path=config_path,
        elastic_modulus=elastic_modulus,
        scaling_factor=scaling_factor,
        poisson_ratio=poisson_ratio,
        window_size=window_size,
        meshsize=meshsize
    )

    # Test: Write TFM results to h5py file
    bytes_hdf5 = io.BytesIO()
    with h5py.File(bytes_hdf5, "w") as h5py_file:
        tfm_results = write_tfm_results(
            h5py_file=h5py_file,
            frame=0,
            pos=np.ndarray([]),
            pos_interp=np.ndarray([]),
            vec_u=np.ndarray([]),
            vec_u_interp=np.ndarray([]),
            vec_f=np.ndarray([]),
            beta=0,
            lamd_opt=0,
            drift_corr_stack=np.ndarray([]),
            cell_img=np.ndarray([]),
            mask=np.ndarray([]),
            config=config.config
        )
    assert isinstance(tfm_results, h5py.File)

    # Test: Read TFM results from example file
    dataset_path = os.path.join("example_data", "example5_tractiondataset", "example_tractiondataset.h5")
    traction_dataset = TractionForceDataset(dataset_path)

    assert list(traction_dataset.columns) == [
        'cell_image',
        'deformation',
        'deformation_interpolated',
        'drift_corrected_stack',
        'frame',
        'mask_roi',
        'noise_beta',
        'optimal_lambda',
        'position',
        'position_interpolated',
        'traction'
    ]
    assert len(traction_dataset) == 2
    assert traction_dataset[0]["frame"][0] == 0
    assert np.array_equal(traction_dataset.frame_indices(), np.array(['0', '1']))
    assert type(traction_dataset.metadata()) is dict
    assert isinstance(traction_dataset[0], pd.core.frame.DataFrame)
    for col in traction_dataset.columns:
        assert isinstance(traction_dataset[col], pd.core.frame.DataFrame)
