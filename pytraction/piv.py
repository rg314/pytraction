import openpiv
import numpy as np
import cv2

from pytraction.utils import allign_slice
from openpiv import tools, pyprocess, validation, filters, scaling, windef, widim



class PIV(object):

    def __init__(self, window_size=32, search_area_size=32, overlap=16, dt=1, scaling_factor = None, settings=None):
        self.scaling_factor = scaling_factor
        self.window_size = window_size
        self.search_area_size = search_area_size
        self.overlap = overlap
        self.dt = dt

        if not settings:
            self._get_default_settings()

    def base_piv(self, img, ref, sig2noise_method='peak2peak'):
        """
        :param img: TFM bead image from stack
        :param ref: Reference frame from TFM

        :return: x, y, u, v displacement field
        """
        u, v, sig2noise = pyprocess.extended_search_area_piv( 
            pyprocess.normalize_intensity(ref), 
            pyprocess.normalize_intensity(img), 
            window_size=self.window_size, 
            overlap=self.overlap, 
            dt=self.dt, 
            search_area_size=self.search_area_size, 
            sig2noise_method=sig2noise_method)

        # prepare centers of the IWs to know where locate the vectors
        x, y = pyprocess.get_coordinates(img.shape, 
                                        search_area_size=self.search_area_size, 
                                        overlap=self.overlap)

        u, v, mask = validation.typical_validation( u, v, 
                                            sig2noise,
                                            settings) 
                                            #threshold = np.percentile(sig2noise,5))

        # removing and filling in the outlier vectors
        u, v = filters.replace_outliers(u, v, method='localmean', 
                                        max_iter=10, 
                                        kernel_size=3)

        # rescale the results to millimeters and mm/sec
        if self.scaling_factor:
            x, y, u, v = scaling.uniform(x, y, u, v, 
                                        scaling_factor=self.scaling_factor)

        # save the data
        x, y, u, v = tools.transform_coordinates(x, y, u, v)

        return x, y, u, v

    @staticmethod
    def _get_default_settings():
        settings = windef.Settings()
        settings.correlation_method='linear'    
        return settings
    
    def iterative_piv(self, img, ref):
        img = allign_slice(img, ref)
        x,y,u,v, mask = widim.WiDIM(ref.astype(np.int32), 
                                    img.astype(np.int32), 
                                    np.ones_like(ref).astype(np.int32), 
                                    min_window_size=self.window_size, 
                                    overlap_ratio=0.5, 
                                    coarse_factor=0, 
                                    dt=self.dt, 
                                    validation_method='mean_velocity', 
                                    trust_1st_iter=0, 
                                    validation_iter=3, 
                                    tolerance=1.5, 
                                    nb_iter_max=1, 
                                    sig2noise_method='peak2peak')
        return x,y,u,v, (img, ref)

