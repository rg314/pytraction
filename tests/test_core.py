from pytraction.core import (
    TractionForceConfig,
    _find_uv_outside_single_polygon,
    _custom_noise,
    _get_noise,
    _write_frame_results,
    _write_metadata_results,
    process_stack,
)

import numpy as np
# from skimage import io
import tifffile
import tempfile
import os
from scipy.ndimage.filters import gaussian_filter
import pickle

def test__custom_noise():
    # dummy config
    class Config():
        def __init__(self) -> None:
            config = None
    
    # fix config / using dummy window size of 32
    config = Config()
    config.config = {'piv': {'min_window_size': 32, 'overlap_ratio': 0.5, 'coarse_factor': 0, 'dt': 1, 'validation_method': 'mean_velocity', 'trust_1st_iter': 0, 'validation_iter': 3, 'tolerance': 1.5, 'nb_iter_max': 1, 'sig2noise_method': 'peak2peak'}, 'tfm': {'E': 'None', 's': 'None', 'meshsize': 'None', 'pix_per_mu': 'None'}, 'settings': {'bead_channel': 0, 'cell_channel': 1, 'segment': False, 'device': 'cpu'}}


    tmppath = os.path.join('tests','data','stack.tiff')

    # test the function
    beta = _custom_noise(tmppath, config)


    # get the temp dir of the cache
    tmpdir = tempfile.gettempdir()
    destination = f'{tmpdir}/tmp_noise.pickle'

    # read in the cached noise
    with open(destination, 'rb') as f:
        cache = pickle.load(f)


    assert os.path.exists(tmppath)
    assert os.path.exists(destination)
    assert beta == 4.010723998246769
    assert cache[tmppath] == beta

    os.remove(destination)

    
def test__find_uv_outside_single_polygon():
    pass
    # _find_uv_outside_single_polygon


test__custom_noise()