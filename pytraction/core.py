import cv2 
from skimage import io
import pandas as pd
import numpy as np
import torch
import segmentation_models_pytorch as smp
from collections import defaultdict
from read_roi import read_roi_file
from shapely import geometry
import pickle

from pytraction.piv import PIV
import pytraction.net.segment as pynet 
from pytraction.utils import normalize, allign_slice, bead_density
from pytraction.traction_force import PyTraction
from pytraction.net.dataloader import get_preprocessing
from pytraction.utils import HiddenPrints


from google_drive_downloader import GoogleDriveDownloader as gdd
import tempfile

class TractionForce(object):

    def __init__(self, scaling_factor, E, s=0.5, meshsize=10, bead_density=None, device='cpu', segment=False, window_size=None):

        self.device = device
        self.segment = segment
        self.window_size = window_size
        self.E = E
        self.s = s

        self.TFM_obj = PyTraction(
            meshsize = meshsize, # grid spacing in pix
            pix_per_mu = scaling_factor,
            E = E, # Young's modulus in Pa
            s = s, # Poisson's ratio
            )

        self.model, self.pre_fn = self.get_model()


    def get_window_size(self, img):
        if not self.window_size:
            density = bead_density(img)

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
            
            window_size = knn.predict([[density]])

            window_size = int(window_size)

            print(f'Automatically selected window size of {window_size}')

            return window_size
        else:
            return self.window_size


    def get_model(self):
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
        if self.device == 'cuda' and torch.cuda.is_available():
            best_model = best_model.to('cuda')
        preproc_fn = smp.encoders.get_preprocessing_fn('efficientnet-b1', 'imagenet')
        preprocessing_fn = get_preprocessing(preproc_fn)

        return best_model, preprocessing_fn


    def get_roi(self, img, ref, frame, roi, img_stack):
        cell_img = np.array(img_stack[frame, 1, :, :])
        cell_img = normalize(cell_img)

        if not roi and self.segment:
            mask = pynet.get_mask(cell_img, self.model, self.pre_fn, device=self.device)

            mask = np.array(mask.astype('bool'), dtype='uint8')

            contours, _ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            areas = [cv2.contourArea(c) for c in contours]
            sorted_areas = np.sort(areas)

            #bounding box (red)
            pts=contours[areas.index(sorted_areas[-1])] #the biggest contour

            cv2.drawContours(cell_img, [pts], -1, (255), 1, cv2.LINE_AA)

            polyx, polyy = np.squeeze(pts, axis=1).T
            roi = True
        
        if roi: 
            shift=0.2
            if not self.segment:
                polyx = roi[0]
                polyy = roi[1]

            minx, maxx = np.min(polyx), np.max(polyx)
            miny, maxy = np.min(polyy), np.max(polyy)

            midx = minx + (maxx-minx) // 2
            midy = miny + (maxy-miny) // 2

            pixel_shift = int(max(midx, midy) * shift) // 2

            # need to raise as an issues
            rescaled = []
            for (xi,yi) in zip(polyx, polyy):
                # apply shift
                rescaled.append([xi, yi])

            self.polygon = geometry.Polygon(rescaled)

            x,y,w,h = cv2.boundingRect(np.array(rescaled))
            pad = 50

            img_crop = img[y-pad:y+h+pad, x-pad:x+w+pad]
            ref_crop = ref[y-pad:y+h+pad, x-pad:x+w+pad]

            if not self.segment:
                pts = np.array(list(zip(polyx, polyy)), np.int32)
                pts = pts.reshape((-1,1,2))
                cv2.polylines(cell_img,[pts],True,(255), thickness=3)

            mask = cv2.fillPoly(np.zeros(cell_img.shape), [pts], (255))
            mask_crop = mask[y-pad:y+h+pad, x-pad:x+w+pad]
            # cnts_crop, _ = cv2.findContours(mask_crop.astype('uint8'),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            # pts_crop = cnts_crop[0]

            cell_img_full = cell_img
            cell_img_crop = cell_img[y-pad:y+h+pad, x-pad:x+w+pad]

            return img_crop, ref_crop, cell_img_crop, mask_crop
        else:
            return img, ref, cell_img, None



    def get_noise(self, x,y,u,v, roi=False):
        if not roi:
            noise = 10
            xn, yn, un, vn = x[:noise],y[:noise],u[:noise],v[:noise]
            noise_vec = np.array([un.flatten(), vn.flatten()])

            varnoise = np.var(noise_vec)
            beta = 1/varnoise
        
        elif roi:
            noise = []
            for (x0,y0, u0, v0) in zip(x.flatten(),y.flatten(), u.flatten(), v.flatten()):
                p1 = geometry.Point([x0,y0])
                if not p1.within(self.polygon):
                    noise.append(np.array([u0, v0]))

            noise_vec = np.array(noise)
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
        img = io.imread(img_path)
        ref = io.imread(ref_path)

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

        else:
            roi = None
        
        if len(img.shape) != 4:
            msg = f'Please ensure that the input image has shape (f,c,w,h) the current shape is {img.shape}'
            raise RuntimeWarning(msg)
        
        if len(ref.shape) != 3:
            msg = f'Please ensure that the input image has shape (f,c,w,h) the current shape is {ref.shape}'
            raise RuntimeWarning(msg)

        return img, ref, roi

    def process_stack(self, img_stack, ref_stack, bead_channel=0, roi=False, frame=[], verbose=1):
        if verbose == 1:
            print('Processing stacks')
            with HiddenPrints():
                output = self._process_stack(img_stack, ref_stack, bead_channel, roi, frame)
        else:
            output = self._process_stack(img_stack, ref_stack, bead_channel, roi, frame)
        return output

    def _process_stack(self, img_stack, ref_stack, bead_channel=0, roi=False, frame=[]):
        nframes = img_stack.shape[0]

        log = defaultdict(list)
        for frame in list(range(nframes)):
            # load plane
            img = normalize(np.array(img_stack[frame, bead_channel, :, :]))
            ref = normalize(np.array(ref_stack[bead_channel,:,:]))

            window_size = self.get_window_size(img)

            
            img_crop, ref_crop, cell_img_crop, mask_crop = self.get_roi(img, ref, frame, roi, img_stack)

            # do piv
            piv_obj = PIV(window_size=window_size)
            x, y, u, v, stack = piv_obj.iterative_piv(img_crop, ref_crop)

            beta = self.get_noise(x,y,u,v, roi=False)

            # make pos and vecs for TFM
            pos = np.array([x.flatten(), y.flatten()])
            vec = np.array([u.flatten(), v.flatten()])

            # compute traction map
            traction_map, f_n_m, L_optimal = self.TFM_obj.calculate_traction_map(pos, vec, beta)

            log['frame'].append(frame)
            log['traction_map'].append(traction_map)
            log['force_field'].append(f_n_m)
            log['stack_bead_roi'].append(stack)
            log['cell_roi'].append(cell_img_crop)
            log['mask_roi'].append(mask_crop)
            log['beta'].append(beta)
            log['L'].append(L_optimal)
            log['pos'].append(pos)
            log['vec'].append(vec)
            log['img_path'].append(self.img_path)
            log['ref_path'].append(self.ref_path)
            log['E'].append(self.E)
            log['s'].append(self.s)

        return pd.DataFrame(log)



