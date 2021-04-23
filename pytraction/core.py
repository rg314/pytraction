import os
import io
import h5py
import pickle
import zipfile 
import tempfile

import cv2 
import skimage
import numpy as np
import pandas as pd
from scipy.spatial import distance


import torch
import segmentation_models_pytorch as smp
from google_drive_downloader import GoogleDriveDownloader as gdd

from shapely import geometry
from read_roi import read_roi_file

import pytraction.net.segment as pynet 
from pytraction.utils import normalize, allign_slice, bead_density, plot, HiddenPrints
from pytraction.process import calculate_traction_map, iterative_piv
from pytraction.net.dataloader import get_preprocessing
from pytraction.dataset import Dataset


class TractionForce(object):

    def __init__(self, scaling_factor, E, min_window_size=None, dt=1, s=0.5, meshsize=10, device='cpu', segment=False, config=None):

        self.device = device
        self.segment = segment
        self.model, self.pre_fn = self._get_cnn_model(device)

        if not config:
            self.config = self._get_config(min_window_size, dt, E, s, meshsize, scaling_factor)
        else:
            self.config = self._config_ymal(config, min_window_size, dt, E, s, meshsize, scaling_factor)

    @staticmethod
    def _get_config(min_window_size, dt, E, s, meshsize, scaling_factor):
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
                        }
                    }
        return config

    @staticmethod
    def _config_ymal(config, min_window_size, dt, E, s, meshsize, scaling_factor):
        config['tfm']['E'] = E,
        config['tfm']['pix_per_mu'] = scaling_factor
        config['tfm']['meshsize'] = meshsize
        config['tfm']['s'] = s
        config['piv']['min_window_size'] = min_window_size
        config['piv']['dt'] = dt
        return config

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


    def get_min_window_size(self, img):
        if not self.config['piv']['min_window_size']:
            density = bead_density(img)

            knn = self._get_knn_model()
            min_window_size = knn.predict([[density]])
            print(f'Automatically selected window size of {min_window_size}')

            return int(min_window_size)
        else:
            return self.config['piv']['min_window_size']


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
    def _located_most_central_cell(counters, mask):
        image_center = np.asarray(mask.shape) / 2
        image_center = tuple(image_center.astype('int32'))

        segmented = []
        for contour in contours:
            # find center of each contour
            M = cv2.moments(contour)
            center_X = int(M["m10"] / M["m00"])
            center_Y = int(M["m01"] / M["m00"])
            contour_center = (center_X, center_Y)
        
            # calculate distance to image_center
            distances_to_center = (distance.euclidean(image_center, contour_center))
        
            # save to a list of dictionaries
            segmented.append({
                'contour': contour, 
                'center': contour_center, 
                'distance_to_center': distances_to_center
                }
                )
    
        sorted_cells = sorted(segmented, key=lambda i: i['distance_to_center'])
        pts=sorted_cells[0]['contour'] #the biggest contour
        return pts

    def _cnn_segment_cell(self, cell_img):
        mask = pynet.get_mask(cell_img, self.model, self.pre_fn, device=self.device)
        return np.array(mask.astype('bool'), dtype='uint8')

    @staticmethod
    def _detect_cell_instances_from_segmentation(mask):
        contours, _ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        return contours

    @staticmethod
    def _predict_roi(self, cell_img):
        # segment image 
        mask = self._cnn_segment_cell(cell_img)
        # get instance outlines 
        contours = self._detect_cell_instances_from_segmentation(mask)
        # find most central cell
        pts = self._located_most_central_cell(counters, mask)
        # get roi
        polyx, polyy = np.squeeze(pts, axis=1).T
        return polyx, polyy

    def _get_polygon_and_roi(self, img_stack, frame, roi):
        cell_img = np.array(img_stack[frame, 1, :, :])
        cell_img = normalize(cell_img)

        if self.segment:
            polyx, polyy = self._predict_roi(cell_img)
            pts = np.array(list(zip(polyx, polyy)), np.int32)
            polygon = geometry.Polygon(pts)
            pts = pts.reshape((-1,1,2))
            return polygon, pts


        elif roi:
            polyx, polyy = roi[0], roi[1]
            pts = np.array(list(zip(polyx, polyy)), np.int32)
            polygon = geometry.Polygon(pts)
            pts = pts.reshape((-1,1,2))
            return polygon, pts
        
        else:
            return None, None

    def _crop_roi(img, ref, frame, img_stack, pts):

            cv2.polylines(cell_img,[pts],True,(255), thickness=3)

            if crop:
                pad = 50
                x,y,w,h = cv2.boundingRect(pts)
                img_crop = img[y-pad:y+h+pad, x-pad:x+w+pad]
                ref_crop = ref[y-pad:y+h+pad, x-pad:x+w+pad]

                mask = cv2.fillPoly(np.zeros(cell_img.shape), [pts], (255))
                mask_crop = mask[y-pad:y+h+pad, x-pad:x+w+pad]

                cell_img_full = cell_img
                cell_img_crop = cell_img[y-pad:y+h+pad, x-pad:x+w+pad]

            else:
                img_crop = img
                ref_crop = ref
                mask_crop = cv2.fillPoly(np.zeros(cell_img.shape), [pts], (255))
                cell_img_crop = cell_img

            return img_crop, ref_crop, cell_img_crop, mask_crop
        else:
            return img, ref, cell_img, None



    def get_noise(self, x,y,u,v, polygon):
        if polygon:
            noise = []
            for (x0,y0, u0, v0) in zip(x.flatten(),y.flatten(), u.flatten(), v.flatten()):
                p1 = geometry.Point([x0,y0])
                if not p1.within(polygon):
                    noise.append(np.array([u0, v0]))

            noise_vec = np.array(noise)
            varnoise = np.var(noise_vec)
            beta = 1/varnoise
        else:
            noise = 10
            xn, yn, un, vn = x[:noise],y[:noise],u[:noise],v[:noise]
            noise_vec = np.array([un.flatten(), vn.flatten()])

            varnoise = np.var(noise_vec)
            beta = 1/varnoise
        return beta


    def _recursive_lookup(self, k, d):
        if k in d: return d[k]
        for v in d.values():
            if isinstance(v, dict):
                a = self._recursive_lookup(k, v)
                if a is not None: return a
        return None

    def load_data(self, img_path, ref_path, roi_path=''):
        """
        :param img_path: Image path for to nd image with shape (f,c,w,h)
        :param ref_path: Reference path for to nd image with shape (c,w,h)
        :param roi_path: 
        """
        img = skimage.io.imread(img_path)
        ref = skimage.io.imread(ref_path)

        if not isinstance(img,np.ndarray) or not isinstance(ref, np.ndarray):
            msg = f'Image data not loaded for {img_path} or {ref_path}'
            raise TypeError(msg)

        if len(img.shape) != 4:
            msg = f'Please ensure that the input image has shape (t,c,w,h) the current shape is {img.shape}'
            raise RuntimeWarning(msg)
        
        if len(ref.shape) != 3:
            msg = f'Please ensure that the input image has shape (c,w,h) the current shape is {ref.shape}'
            raise RuntimeWarning(msg)


        # messy fix to include file name in log file
        self.ref_path = ref_path
        self.img_path = img_path

        if '.csv' in roi_path:
            x, y = pd.read_csv(roi_path).T.values
            roi = (x,y)

        elif '.roi' in roi_path:
            d = read_roi_file(roi_path)
            x = self._recursive_lookup('x', d)
            y = self._recursive_lookup('y', d)

            roi = (x,y)

        elif '.zip' in roi_path:
            rois = []
            with zipfile.ZipFile(roi_path) as ziproi:
                for file in ziproi.namelist():
                    roi_path_file = ziproi.extract(file)
                    d = read_roi_file(roi_path_file)
                    x = self._recursive_lookup('x', d)
                    y = self._recursive_lookup('y', d)
                    rois.append((x,y))
                    os.remove(roi_path_file)
            
            roi = rois


        else:
            roi = None
        

        return img, ref, roi

    def process_stack(self, img_stack, ref_stack, bead_channel=0, roi=False, frame=[], crop=False, verbose=0):
        if verbose == 0:
            print('Processing stacks')
            with HiddenPrints():
                output = self._process_stack(img_stack, ref_stack, bead_channel, roi, frame, crop)
        elif verbose == 1:
            output = self._process_stack(img_stack, ref_stack, bead_channel, roi, frame, crop)
        return output

    def _process_stack(self, img_stack, ref_stack, bead_channel=0, roi=False, frame=[], crop=False):
        nframes = img_stack.shape[0]

        bytes_hdf5 = io.BytesIO()

        with h5py.File(bytes_hdf5, 'w') as results:

            for frame in list(range(nframes)):
                # load plane
                img = normalize(np.array(img_stack[frame, bead_channel, :, :]))
                ref = normalize(np.array(ref_stack[bead_channel,:,:]))

                min_window_size = self.get_min_window_size(img)
                self.config['piv']['min_window_size'] = min_window_size

                if isinstance(roi, list):
                    assert len(roi) == nframes, f'Warning ROI list has len {len(roi)} which is not equal to \
                                                the number of frames ({nframes}). This would suggest that you do not \
                                                have the correct number of ROIs in the zip file.'
                    roi_i = roi[frame]
                else:
                    roi_i = roi

                
                img_crop, ref_crop, cell_img_crop, mask_crop = self.get_roi(img, ref, frame, roi_i, img_stack, crop)

                # do piv
                x, y, u, v, stack = iterative_piv(img_crop, ref_crop, config=self.config)

                beta = self.get_noise(x,y,u,v, roi=False)

                # make pos and vecs for TFM
                pos = np.array([x.flatten(), y.flatten()])
                vec = np.array([u.flatten(), v.flatten()])

                # compute traction map
                traction_map, f_n_m, L_optimal = calculate_traction_map(pos, vec, beta, 
                    self.config['tfm']['meshsize'], 
                    self.config['tfm']['s'], 
                    self.config['tfm']['pix_per_mu'], 
                    self.config['tfm']['E']
                    )

                results[f'frame/{frame}'] = frame
                results[f'traction_map/{frame}'] = traction_map
                results[f'force_field/{frame}'] = f_n_m
                results[f'stack_bead_roi/{frame}'] = stack
                results[f'cell_roi/{frame}'] = cell_img_crop
                results[f'mask_roi/{frame}'] = 0 if mask_crop is None else mask_crop
                results[f'beta/{frame}'] = beta
                results[f'L/{frame}'] = L_optimal
                results[f'pos/{frame}'] = pos
                results[f'vec/{frame}'] = vec
            
            # create metadata with a placeholder
            results['metadata'] = 0

            for k,v in self.config['piv'].items():
                results['metadata'].attrs[k] = np.void(str(v).encode())

            for k,v in self.config['tfm'].items():
                results['metadata'].attrs[k] = np.void(str(v).encode())

            # to recover
            # h5py.File(results)['metadata'].attrs['img_path'].tobytes()

        return Dataset(bytes_hdf5)



