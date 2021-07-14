from scipy.ndimage.measurements import label
from pytraction.core import (
    TractionForceConfig,
    _find_uv_outside_single_polygon,
    _custom_noise,
    _get_noise,
    _write_frame_results,
    _write_metadata_results,
    process_stack,
)

import os
import tempfile
import pickle
import numpy as np
from shapely.geometry import Polygon

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

    # function to create circle with radius at center (x,y)
    def create_circle(radius, center=(10,10)):
        theta = np.linspace(0, 2*np.pi, 30)
        x = radius*np.cos(theta) + center[0]
        y = radius*np.sin(theta) + center[1]
        return x, y

    
    # create 3 three consentic circles with increasing radii
    inner_x, inner_y = create_circle(2)
    polygon_x, polygon_y = create_circle(4)
    outer_x, outer_y = create_circle(6)

    # create a polygon with middle circle
    polygon = Polygon(list(zip(polygon_x, polygon_y)))

    # create random u and v vectors
    u_inner = np.random.randint(0,100,size=30)
    u_outer = u_inner + 10
    v_inner = np.random.randint(0,100,size=30)
    v_outer = v_inner + 10

    # join inner and outer circle (x,y,u,v)
    x = np.concatenate([inner_x, outer_x])
    y = np.concatenate([inner_y, outer_y])
    u = np.concatenate([u_inner, u_outer])
    v = np.concatenate([v_inner, v_outer])

    # test function
    pts = _find_uv_outside_single_polygon(x, y, u, v, polygon)

    # get the known input
    pts_test = np.array([u_outer, v_outer]).T

    assert (pts_test==pts).all()


    _find_uv_outside_single_polygon(np.ndarray(1), np.ndarray(1), u, v, polygon)



test__find_uv_outside_single_polygon()