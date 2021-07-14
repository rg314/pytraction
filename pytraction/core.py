import os
import io
from typing import Tuple, Type
import h5py
import pickle
import tempfile
from numpy.lib.npyio import load

import tifffile
import numpy as np
from shapely import geometry

import torch
import segmentation_models_pytorch as smp
from google_drive_downloader import GoogleDriveDownloader as gdd
import yaml


from pytraction.preprocess import _get_raw_frames, _get_min_window_size, _get_polygon_and_roi, _create_crop_mask_targets, _load_frame_roi
from pytraction.process import calculate_traction_map, iterative_piv
from pytraction.net.dataloader import get_preprocessing
from pytraction.dataset import Dataset
from pytraction.utils import normalize
from pytraction.roi import roi_loaders


class TractionForceConfig(object):

    def __init__(
        self, 
        E:float, 
        scaling_factor:float, 
        config:str, 
        min_window_size:int=None, 
        meshsize:int=10, 
        s:float=0.5, 
        knn:bool=True,
        cnn:bool=True,
    ):

        self.config = self._config_yaml(config, E, min_window_size, s, meshsize, scaling_factor)

        
        self.knn = self._get_knn_model() if knn else None
        self.model, self.pre_fn = self._get_cnn_model(device=self.config['settings']['device']) if cnn else (None, None)


    def __repr__():
        pass

    @staticmethod
    def _config_yaml(config, E, min_window_size, s, meshsize, scaling_factor):
        with open(config, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        
        config['tfm']['E'] = E,
        config['tfm']['pix_per_mu'] = scaling_factor
        config['piv']['min_window_size'] = min_window_size if min_window_size is not None else config['piv']['min_window_size']
        config['tfm']['s'] = s if s is not None else config['tfm']['s']
        config['tfm']['meshsize'] = meshsize if meshsize is not None else config['tfm']['meshsize']
        return config


    @staticmethod
    def _get_cnn_model(device):
        # data_20210320.zip
        file_id = '1zShYcG8IMsMjB8hA6FcBTIZPfi_wDL4n'
        tmpdir = tempfile.gettempdir()
        destination = f'{tmpdir}/model.zip'


        gdd.download_file_from_google_drive(file_id=file_id,
                                        dest_path=destination,
                                        unzip=True,
                                        showsize=True,
                                        overwrite=False)


        # currently using model from 20210316
        best_model = torch.load(f'{tmpdir}/best_model_1.pth', map_location='cpu')
        if device == 'cuda' and torch.cuda.is_available():
            best_model = best_model.to('cuda')
        preproc_fn = smp.encoders.get_preprocessing_fn('efficientnet-b1', 'imagenet')
        preprocessing_fn = get_preprocessing(preproc_fn)

        return best_model, preprocessing_fn

    @staticmethod
    def _get_knn_model():
        file_id = '1xQuGSUdW3nIO5lAm7DQb567sMEQgHmQD'
        tmpdir = tempfile.gettempdir()
        destination = f'{tmpdir}/knn.zip'


        gdd.download_file_from_google_drive(file_id=file_id,
                                        dest_path=destination,
                                        unzip=True,
                                        showsize=False,
                                        overwrite=False)

        with open(f'{tmpdir}/knn.pickle', 'rb') as f:
            knn = pickle.load(f)
        
        return knn


    def load_data(self, img_path:str, ref_path:str, roi_path:str='')-> Tuple[np.ndarray, np.ndarray, list]:
        """[summary]

        Args:
            img_path (str): Image path for to nd image with shape (f,c,w,h)
            ref_path (str): Reference path for to nd image with shape (c,w,h)
            roi_path (str, optional): [description]. Defaults to ''.

        Raises:
            TypeError: Image data not loaded for img or ref path. Use .tif as ext'
            RuntimeWarning: Please ensure that the input image has shape (t,c,w,h)
            RuntimeWarning: Please ensure that the input ref image has shape (c,w,h)

        Returns:
            Tuple[np.ndarray, np.ndarray, list]: [description]
        """
        img = tifffile.imread(img_path)
        ref = tifffile.imread(ref_path)
        roi = roi_loaders(roi_path)


        if not isinstance(img,np.ndarray) or not isinstance(ref, np.ndarray):
            msg = f'Image data not loaded for {img_path} or {ref_path}'
            raise TypeError(msg)

        if len(img.shape) != 4:
            msg = f'Please ensure that the input image has shape (t,c,w,h) the current shape is {img.shape}'
            raise RuntimeWarning(msg)
        
        if len(ref.shape) != 3:
            msg = f'Please ensure that the input ref image has shape (c,w,h) the current shape is {ref.shape}'
            raise RuntimeWarning(msg)

        return img, ref, roi



def _find_uv_outside_single_polygon(x:np.ndarray,y:np.ndarray,u:np.ndarray,v:np.ndarray, polygon:Type[geometry.Polygon]) -> np.ndarray:
    """Find the u and v components outside the ROI polygon. 

    Args:
        x (np.ndarray): x-component
        y (np.ndarray): y-component
        u (np.ndarray): u-component
        v (np.ndarray): v-component
        polygon (Type[geometry.Polygon]): shapely polygon to test which (xi, yi) is within

    Returns:
        np.ndarray: (un, vn) array with noisy u and v components
    """
    noise = []
    for (x0,y0, u0, v0) in zip(x.flatten(),y.flatten(), u.flatten(), v.flatten()):        
        p1 = geometry.Point([x0,y0])
        if not p1.within(polygon):
            noise.append(np.array([u0, v0]))
    return np.array(noise)


def _custom_noise(tiff_stack:np.ndarray, config:Type[TractionForceConfig]) -> float:
    """Returns the value for beta of custom noise provided as a tiff stack.

    Args:
        tiff_stack (np.ndarray): tiff stack with shape (t,w,h)
        config (Type[TractionForceConfig]): Configuration for traction object.

    Returns:
        float: beta which is 1/var(u,v)
    """

    tmpdir = tempfile.gettempdir()
    destination = f'{tmpdir}/tmp_noise.pickle'
    cache = dict()

    if os.path.exists(destination):
        with open(destination, 'rb') as f:
            cache = pickle.load(f)
        beta = cache.get(tiff_stack, None)

        if beta:
            return beta

    tiff_noise_stack = tifffile.imread(tiff_stack)
    un, vn = np.array([]), np.array([])
    max_range = max(tiff_noise_stack.shape[0]-1, 3-1)
    for i in range(max_range):
        img = normalize(tiff_noise_stack[i,:,:])
        ref = normalize(tiff_noise_stack[i+1,:,:])
        x, y, u, v, stack = iterative_piv(img, ref, config)
        un = np.append(un, u)
        vn = np.append(vn, v)

    noise_vec = np.array([un.flatten(), vn.flatten()])
    varnoise = np.var(noise_vec)
    beta = 1/varnoise
    cache[tiff_stack] = beta

    with open(destination, 'wb') as f:
        pickle.dump(cache, f)


    return beta

def _get_noise(config, x=None,y=None,u=None,v=None, polygon=None, custom_noise=None):
    if polygon:
        noise_vec = _find_uv_outside_single_polygon(x,y,u,v, polygon)
    elif custom_noise:
        return _custom_noise(custom_noise, config)
    else:
        noise = 10
        xn, yn, un, vn = x[:noise],y[:noise],u[:noise],v[:noise]
        noise_vec = np.array([un.flatten(), vn.flatten()])

    varnoise = np.var(noise_vec)
    beta = 1/varnoise
    return beta

def _write_frame_results(results, frame, traction_map, f_n_m, stack, cell_img, mask, beta, L_optimal, pos, vec):
    results[f'frame/{frame}'] = frame
    results[f'traction_map/{frame}'] = traction_map
    results[f'force_field/{frame}'] = f_n_m
    results[f'stack_bead_roi/{frame}'] = stack
    results[f'cell_roi/{frame}'] = cell_img
    results[f'mask_roi/{frame}'] = 0 if mask is None else mask
    results[f'beta/{frame}'] = beta
    results[f'L/{frame}'] = L_optimal
    results[f'pos/{frame}'] = pos
    results[f'vec/{frame}'] = vec
    return results

def _write_metadata_results(results, config):
    # create metadata with a placeholder
    results['metadata'] = 0

    for k,v in config['piv'].items():
        results['metadata'].attrs[k] = np.void(str(v).encode())

    for k,v in config['tfm'].items():
        results['metadata'].attrs[k] = np.void(str(v).encode())
    return results

def process_stack(img_stack, ref_stack, config, bead_channel=0, cell_channel=1, roi=False, frame=[], crop=False, verbose=0, custom_noise=None):
    nframes = img_stack.shape[0]
    

    bytes_hdf5 = io.BytesIO()

    with h5py.File(bytes_hdf5, 'w') as results:

        for frame in list(range(nframes)):
            # load planes
            img, ref, cell_img = _get_raw_frames(img_stack, ref_stack, frame, bead_channel, cell_channel)

            # get_minimum window_size
            min_window_size = _get_min_window_size(img, config)
            config.config['piv']['min_window_size'] = min_window_size

            # load_rois
            roi_i = _load_frame_roi(roi, frame, nframes)

            # compute polygon and roi 
            polygon, pts = _get_polygon_and_roi(cell_img, roi_i, config)

            # crop targets
            img, ref, cell_img, mask = _create_crop_mask_targets(img, ref, cell_img, pts, crop, pad=50)

            # do PIV
            x, y, u, v, (stack, dx, dy) = iterative_piv(img, ref, config)

            # calculate noise 
            beta = _get_noise(config, x,y,u,v, polygon, custom_noise=custom_noise)

            # make pos and vecs for TFM
            pos = np.array([x.flatten(), y.flatten()])
            vec = np.array([u.flatten(), v.flatten()])

            # compute traction map
            traction_map, f_n_m, L_optimal = calculate_traction_map(pos, vec, beta, 
                config.config['tfm']['meshsize'], 
                config.config['tfm']['s'], 
                config.config['tfm']['pix_per_mu'], 
                config.config['tfm']['E']
                )

            # write results for frame to h5
            results = _write_frame_results(results, frame, traction_map, f_n_m, stack, cell_img, mask, beta, L_optimal, pos, vec)

        # write metadata to results
        results = _write_metadata_results(results, config.config)

        # to recover
        # h5py.File(results)['metadata'].attrs['img_path'].tobytes()

    return Dataset(bytes_hdf5)
