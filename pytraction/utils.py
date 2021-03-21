import os 
import sys

from scipy.sparse import linalg as splinalg
import scipy.sparse as sparse

import cv2
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable


def allign_slice(img, ref):
    """
    :param img: Image slice to allign
    :param ref: Reference bead to calculate alignment

    :return: Aligned image
    """
    # amount to reduce template
    depth = int(min(img.shape)*0.1)

    # calculate matchTemplate using ccorr_normed method
    tm_ccorr_normed = cv2.matchTemplate(img,ref[depth:-depth, depth:-depth],cv2.TM_CCORR_NORMED)
    max_ccorr = np.unravel_index(np.argmax(tm_ccorr_normed, axis=None), tm_ccorr_normed.shape)

    # shifts in the x and y
    dy = depth - max_ccorr[0]
    dx = depth - max_ccorr[1]

    # transformation matrix
    rows,cols = img.shape
    M = np.float32([[1,0,dx],[0,1,dy]])
    return cv2.warpAffine(img,M,(cols,rows))


def sparse_cholesky(A): # The input matrix A must be a sparse symmetric positive-definite.
    n = A.shape[0]
    LU = splinalg.splu(A,diag_pivot_thresh=0) # sparse LU decomposition
    
    return LU.L.dot( sparse.diags(LU.U.diagonal()**0.5) )


def normalize(x):
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return np.array(x*255, dtype='uint8')



def plot(log, frame=0, vmax=None):
    traction_map = log['traction_map'][frame]
    cell_roi = log['cell_roi'][frame]
    x, y = log['pos'][frame]
    u, v = log['vec'][frame]
    L = log['L'][frame]
    vmax = np.max(traction_map) if not vmax else vmax
 
    
    fig, ax = plt.subplots(1,2)
    im1 = ax[0].imshow(traction_map, interpolation='bicubic', cmap='jet',extent=[x.min(), x.max(), y.min(), y.max()], vmin=0, vmax=vmax)
    ax[0].quiver(x, y, u, v)
    divider1 = make_axes_locatable(ax[0])
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)

    im2 = ax[1].imshow(cell_roi, cmap='gray',vmax=np.max(cell_roi))

    fig.colorbar(im1, cax=cax1)

    ax[0].set_axis_off()
    ax[1].set_axis_off()
    plt.tight_layout()
    

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

