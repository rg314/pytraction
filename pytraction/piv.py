import openpiv
import numpy as np
import cv2
from openpiv import tools, pyprocess, validation, filters, scaling 


class PIV(object):

    def __init__(self, window_size=64, search_area_size=64, overlap=32, dt=1, scaling_factor = None):
        self.scaling_factor = scaling_factor
        self.window_size = window_size
        self.search_area_size = search_area_size
        self.overlap = overlap
        self.dt = dt

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

        u, v, mask = validation.sig2noise_val( u, v, 
                                            sig2noise, 
                                            threshold = np.percentile(sig2noise,5))

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




    # for x in [256, 128, 64, 32, 16]:
    #     x, y, u, v = base_piv(frame_a, frame_b, window_size=x, search_area_size=x, overlap=x//2)
    #     m = np.sqrt(u**2 + v**2).flatten()
    #     fig, ax = plt.subplots(1,2)
    #     ax[0].hist(m)
    #     ax[1].imshow(frame_a, cmap='gray')
    #     ax[1].quiver(x,y, u,v)
    #     plt.show()