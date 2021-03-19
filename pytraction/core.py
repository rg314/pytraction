import cv2 
import numpy as np

from pytraction.piv import PIV
import pytraction.net.segment as pynet 
from pytraction.traction_force import PyTraction



class TractionForce(object):

    def __init__(self, scaling_factor, E,roi=False, s=0.5, meshsize=10, bead_density=None, device='cpu', segment=True):

        self.device = device

        self.TFM_obj = PyTraction(
            meshsize = meshsize, # grid spacing in pix
            pix_per_mu = scaling_factor,
            E = E, # Young's modulus in Pa
            s = s, # Poisson's ratio
            )

        self.window_size = self.get_windows_size(bead_density)


        self.PIV_obj = PIV(window_size=self.window_size)
        self.model, self.pre_fn = self.get_model(self.device)
        self.roi = roi

    @staticmethod
    def get_windows_size(bead_density):
        if not bead_density:
            return 16
        else:
            return 16

    def get_model(self):
        # currently using model from 20210316
        best_model = torch.hub.load_state_dict_from_url(
            'https://docs.google.com/uc?export=download&id=1zShYcG8IMsMjB8hA6FcBTIZPfi_wDL4n', map_location='cpu')
        if self.device == 'cuda' and torch.cuda.is_available():
            best_model = best_model.to('cuda')
        preproc_fn = smp.encoders.get_preprocessing_fn('efficientnet-b1', 'imagenet')
        preprocessing_fn = get_preprocessing(preproc_fn)

        return best_model, preprocessing_fn


    def get_roi(self, img, roi):

        if not self.roi and self.segment:
            cell_img = np.array(img1[frame, 1, :, :])
            cell_img = normalize(cell_img)
            mask = pynet.get_mask(cell_img, self.model, self.pre_fn, device=self.device)

            mask = np.array(~mask.astype('bool'), dtype='uint8')

            contours, _ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            areas = [cv2.contourArea(c) for c in contours]
            sorted_areas = np.sort(areas)

            #bounding box (red)
            cnt=contours[areas.index(sorted_areas[-1])] #the biggest contour

            cv2.drawContours(cell_img, [cnt], -1, (255), 1, cv2.LINE_AA)

            polyx, polyy = np.squeeze(cnt, axis=1).T
            roi = True
        
        if self.roi: 
            shift=0.2
            if not segment:
                polyx = roi[name.split('.')[0]]['x']
                polyy = roi[name.split('.')[0]]['y']

            minx, maxx = np.min(polyx), np.max(polyx)
            miny, maxy = np.min(polyy), np.max(polyy)

            midx = minx + (maxx-minx) // 2
            midy = miny + (maxy-miny) // 2

            pixel_shift = int(max(midx, midy) * shift) // 2

            rescaled = []
            for (xi,yi) in zip(polyx, polyy):
                if xi < midx:
                    x_shift = xi - pixel_shift
                else:
                    x_shift = xi + pixel_shift

                if yi < midy:
                    y_shift = yi - pixel_shift
                else:
                    y_shift = yi + pixel_shift
                
                rescaled.append([x_shift, y_shift])


            polygon = geometry.Polygon(rescaled)

            x,y,w,h = cv2.boundingRect(np.array(rescaled))
            pad = 50

            img = img[y-pad:y+h+pad, x-pad:x+w+pad]
            ref = ref[y-pad:y+h+pad, x-pad:x+w+pad]

            if not segment:
                cell_img = np.array(img1[frame, 1, :, :])
                pts = np.array(list(zip(polyx, polyy)), np.int32)
                pts = pts.reshape((-1,1,2))
                cv2.polylines(cell_img,[pts],True,(255), thickness=3)
            
            cell_img_full = cell_img
            cell_img = cell_img[y-pad:y+h+pad, x-pad:x+w+pad]


    def get_noise(self):

        if not self.roi:
            noise = 10
            xn, yn, un, vn = x[:noise],y[:noise],u[:noise],v[:noise]
            noise_vec = np.array([un.flatten(), vn.flatten()])

            varnoise = np.var(noise_vec)
            beta = 1/varnoise
        
        elif self.roi:
            noise = []
            for (x0,y0, u0, v0) in zip(x.flatten(),y.flatten(), u.flatten(), v.flatten()):
                p1 = geometry.Point([x0,y0])
                if not p1.within(polygon):
                    noise.append(np.array([u0, v0]))

            noise_vec = np.array(noise)
            varnoise = np.var(noise_vec)
            beta = 1/varnoise
        return beta



    def process_frame():
        pass 

    def process_stack():
        x, y, u, v, stack = piv_obj.iterative_piv(img, ref, tmp)