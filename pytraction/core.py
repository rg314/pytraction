import os
import io
import h5py
import pickle
import zipfile 
import tempfile
from typing import Type, Tuple, Union, IO
from read_roi import read_roi_file

import skimage
import numpy as np
import pandas as pd
from shapely import geometry

import torch
import segmentation_models_pytorch as smp
from google_drive_downloader import GoogleDriveDownloader as gdd


from pytraction.preprocess import _get_raw_frames, _get_min_window_size, _get_polygon_and_roi, _create_crop_mask_targets, _load_frame_roi
from pytraction.process import calculate_traction_map, iterative_piv
from pytraction.net.dataloader import get_preprocessing
from pytraction.dataset import Dataset


class TractionForceConfig(object):

    def __init__(self, scaling_factor: float, E: float, min_window_size=None, dt=1, s=0.5, meshsize=10, device='cpu', segment=False, config=None):
        """[summary]

        Args:
            scaling_factor (float): [description]
            E (float): [description]
            min_window_size ([type], optional): [description]. Defaults to None.
            dt (int, optional): [description]. Defaults to 1.
            s (float, optional): [description]. Defaults to 0.5.
            meshsize (int, optional): [description]. Defaults to 10.
            device (str, optional): [description]. Defaults to 'cpu'.
            segment (bool, optional): [description]. Defaults to False.
            config ([type], optional): [description]. Defaults to None.
        """

        self.device = device
        self.model, self.pre_fn = self._get_cnn_model(device)
        self.knn = self._get_knn_model()

        if not config:
            self.config = self._get_config(min_window_size, dt, E, s, meshsize, scaling_factor, segment)
        else:
            self.config = self._config_ymal(config, min_window_size, dt, E, s, meshsize, scaling_factor)


    def __repr__():
        pass

    @staticmethod
    def _get_config(min_window_size: int, dt: float, E: float, s: float, 
        meshsize: int, scaling_factor: float, segment: bool) -> dict:
        """[summary]

        Args:
            min_window_size (int): [description]
            dt (float): [description]
            E (float): [description]
            s (float): [description]
            meshsize (int): [description]
            scaling_factor (float): [description]
            segment (bool): [description]

        Returns:
            dict: [description]
        """

        config = {
                'piv':{
                    'min_window_size':min_window_size, 
                    'overlap_ratio':0.5, 
                    'coarse_factor':0, 
                    'dt':dt, 
                    'validation_method':'mean_velocity', 
                    'trust_1st_iter':0, 
                    'validation_iter':3, 
                    'tolerance':1.5, 
                    'nb_iter_max':1, 
                    'sig2noise_method':'peak2peak',
                    },
                'tfm': {
                    'E':E,
                    's':s,
                    'meshsize':meshsize,
                    'pix_per_mu':scaling_factor
                        },

                'settings': {
                    'segment':segment
                        },
                    }
        return config

    @staticmethod
    def _config_ymal(config: dict, min_window_size: int, dt: float, 
        E:float, s:float, meshsize: int, scaling_factor:float) -> dict:
        """[summary]

        Args:
            config (dict): [description]
            min_window_size (int): [description]
            dt (float): [description]
            E (float): [description]
            s (float): [description]
            meshsize (int): [description]
            scaling_factor (float): [description]

        Returns:
            dict: [description]
        """

        config['tfm']['E'] = E,
        config['tfm']['pix_per_mu'] = scaling_factor
        config['tfm']['meshsize'] = meshsize
        config['tfm']['s'] = s
        config['piv']['min_window_size'] = min_window_size
        config['piv']['dt'] = dt
        return config


    @staticmethod
    def _get_cnn_model(device: str) -> tuple:
        """[summary]

        Args:
            device (str): [description]

        Returns:
            tuple: [description]
        """

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
        """[summary]

        Returns:
            [type]: [description]
        """

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


    def _recursive_lookup(self, k:str, d:dict) -> list:
        """[summary]

        Args:
            k (str): [description]
            d (dict): [description]

        Returns:
            list: [description]
        """
        if k in d: return d[k]
        for v in d.values():
            if isinstance(v, dict):
                a = self._recursive_lookup(k, v)
                if a is not None: return a
        return None


    @staticmethod
    def _load_csv_roi(roi_path: str) -> tuple:
        """[summary]

        Args:
            roi_path (str): [description]

        Returns:
            tuple: [description]
        """
        x, y = pd.read_csv(roi_path).T.values
        return (x,y)

    def _load_roireader_roi(self, roi_path):
        d = read_roi_file(roi_path)
        x = self._recursive_lookup('x', d)
        y = self._recursive_lookup('y', d)

        return (x,y)

    def _load_zip_roi(self, roi_path: str) -> list:
        """[summary]

        Args:
            roi_path (str): [description]

        Returns:
            list: [description]
        """
        rois = []
        with zipfile.ZipFile(roi_path) as ziproi:
            for file in ziproi.namelist():
                roi_path_file = ziproi.extract(file)
                d = read_roi_file(roi_path_file)
                x = self._recursive_lookup('x', d)
                y = self._recursive_lookup('y', d)
                rois.append((x,y))
                os.remove(roi_path_file)
        return rois

    def _roi_loaders(self, roi_path: str) -> Union[tuple, list, None]:
        """[summary]

        Args:
            roi_path (str): [description]

        Returns:
            Union[tuple, list, None]: [description]
        """
        if '.csv' in roi_path:
            return self._load_csv_roi(roi_path)

        elif '.roi' in roi_path:
            return self._load_roireader_roi(roi_path)

        elif '.zip' in roi_path:
            return self._load_zip_roi(roi_path)

        else:
            return None


    def load_data(self, img_path:str, ref_path:str, roi_path='') -> Tuple[np.ndarray, np.ndarray, tuple]:
        """[summary]

        Args:
            img_path (str): [description]
            ref_path (str): [description]
            roi_path (str, optional): [description]. Defaults to ''.

        Raises:
            TypeError: [description]
            RuntimeWarning: [description]
            RuntimeWarning: [description]

        Returns:
            Tuple[np.ndarray, np.ndarray, tuple]: [description]
        """
        img = skimage.io.imread(img_path)
        ref = skimage.io.imread(ref_path)
        roi = self._roi_loaders(roi_path)


        if not isinstance(img,np.ndarray) or not isinstance(ref, np.ndarray):
            msg = f'Image data not loaded for {img_path} or {ref_path}'
            raise TypeError(msg)

        if len(img.shape) != 4:
            msg = f'Please ensure that the input image has shape (t,c,w,h) the current shape is {img.shape}'
            raise RuntimeWarning(msg)
        
        if len(ref.shape) != 3:
            msg = f'Please ensure that the input image has shape (c,w,h) the current shape is {ref.shape}'
            raise RuntimeWarning(msg)

        return img, ref, roi

def _find_uv_outside_single_polygon(x:np.ndarray,y:np.ndarray,u:np.ndarray,v:np.ndarray, 
    polygon:np.ndarray) -> np.ndarray:
    """[summary]

    Args:
        x (np.ndarray): [description]
        y (np.ndarray): [description]
        u (np.ndarray): [description]
        v (np.ndarray): [description]
        polygon (np.ndarray): [description]

    Returns:
        np.ndarray: [description]
    """
    noise = []
    for (x0,y0, u0, v0) in zip(x.flatten(),y.flatten(), u.flatten(), v.flatten()):        
        p1 = geometry.Point([x0,y0])
        if not p1.within(polygon):
            noise.append(np.array([u0, v0]))
    return np.array(noise)

def _get_noise(x:np.ndarray,y:np.ndarray,u:np.ndarray,v:np.ndarray, polygon:np.ndarray) -> float:
    """[summary]

    Args:
        x (np.ndarray): [description]
        y (np.ndarray): [description]
        u (np.ndarray): [description]
        v (np.ndarray): [description]
        polygon (np.ndarray): [description]

    Returns:
        float: [description]
    """
    if polygon:
        noise_vec = _find_uv_outside_single_polygon(x,y,u,v, polygon)
    else:
        noise = 10
        xn, yn, un, vn = x[:noise],y[:noise],u[:noise],v[:noise]
        noise_vec = np.array([un.flatten(), vn.flatten()])

    varnoise = np.var(noise_vec)
    beta = 1/varnoise
    return beta

def _write_frame_results(results:IO[bytes], frame:int, traction_map:np.ndarray, f_n_m:np.ndarray, 
    stack:np.ndarray, cell_img:np.ndarray, mask:np.ndarray, beta:float, 
    L_optimal:float, pos:np.ndarray, vec:np.ndarray) -> IO[bytes]:
    """[summary]

    Args:
        results (IO[bytes]): [description]
        frame (int): [description]
        traction_map (np.ndarray): [description]
        f_n_m (np.ndarray): [description]
        stack (np.ndarray): [description]
        cell_img (np.ndarray): [description]
        mask (np.ndarray): [description]
        beta (float): [description]
        L_optimal (float): [description]
        pos (np.ndarray): [description]
        vec (np.ndarray): [description]

    Returns:
        IO[bytes]: [description]
    """

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

def _write_metadata_results(results:IO[bytes], config: Type['TractionForceConfig']) -> IO[bytes]:
    """[summary]

    Args:
        results (IO[bytes]): [description]
        config (Type[): [description]

    Returns:
        IO[bytes]: [description]
    """
    # create metadata with a placeholder
    results['metadata'] = 0

    for k,v in config['piv'].items():
        results['metadata'].attrs[k] = np.void(str(v).encode())

    for k,v in config['tfm'].items():
        results['metadata'].attrs[k] = np.void(str(v).encode())
    return results

def process_stack(img_stack:np.ndarray, ref_stack:np.ndarray, config:Type['TractionForceConfig'], 
    bead_channel=0, cell_channel=1, roi=False, frame=[], crop=False, verbose=0) -> Type[Dataset]:
    """[summary]

    Args:
        img_stack (np.ndarray): [description]
        ref_stack (np.ndarray): [description]
        config (Type[): [description]
        bead_channel (int, optional): [description]. Defaults to 0.
        cell_channel (int, optional): [description]. Defaults to 1.
        roi (bool, optional): [description]. Defaults to False.
        frame (list, optional): [description]. Defaults to [].
        crop (bool, optional): [description]. Defaults to False.
        verbose (int, optional): [description]. Defaults to 0.

    Returns:
        Type[Dataset]: [description]
    """
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
            x, y, u, v, stack = iterative_piv(img, ref, config)

            # calculate noise 
            beta = _get_noise(x,y,u,v, polygon)

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
