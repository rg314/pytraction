import os 
import sys

import scipy.sparse as sparse
from scipy.interpolate import griddata
from scipy.sparse import linalg as splinalg

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
    return dx, dy, cv2.warpAffine(img,M,(cols,rows))


def sparse_cholesky(A): # The input matrix A must be a sparse symmetric positive-definite.
    n = A.shape[0]
    LU = splinalg.splu(A.tocsc(),diag_pivot_thresh=0) # sparse LU decomposition
    
    return LU.L.dot( sparse.diags(LU.U.diagonal()**0.5)).tocsr()


def interp_vec2grid(pos, vec, cluster_size, grid_mat=np.array([])):
    if not grid_mat:
        max_eck = [np.max(pos[0]), np.max(pos[1])]
        min_eck = [np.min(pos[0]), np.min(pos[1])]

        i_max = np.floor((max_eck[0]-min_eck[0])/cluster_size)
        j_max = np.floor((max_eck[1]-min_eck[1])/cluster_size)
        
        i_max = i_max - np.mod(i_max,2)
        j_max = j_max - np.mod(j_max,2)

        X = min_eck[0] + np.arange(0.5, i_max)*cluster_size
        Y = min_eck[1] + np.arange(0.5, j_max)*cluster_size

        x, y = np.meshgrid(X, Y)

        grid_mat = np.stack([x,y], axis=2)

        u = griddata(pos.T, vec.T, (x,y),method='cubic')

        return grid_mat,u, int(i_max), int(j_max)

def normalize(x):
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return np.array(x*255, dtype='uint8')

def clahe(data):
    img = cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)[:,:,0]


def bead_density(img):
    clahe_img = clahe(normalize(img))
    norm = cv2.adaptiveThreshold(clahe_img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,5,2)/255
    
    ones = len(norm[norm == 1])
    
    area = img.shape[0]* img.shape[1]
    area_beads = ones/area
    
    return area_beads


def plot(log, frame=0, vmax=None, mask=True, figsize=(16,16)):
    log = log[frame]
    traction_map = log['traction_map'][0]
    cell_roi = log['cell_roi'][0]
    x, y = log['pos'][0]
    u, v = log['vec'][0]
    L = log['L'][0]
    vmax = np.max(traction_map) if not vmax else vmax

    fig, ax = plt.subplots(1,2, figsize=figsize)
    im1 = ax[0].imshow(traction_map, interpolation='bicubic', cmap='jet',extent=[x.min(), x.max(), y.min(), y.max()], vmin=0, vmax=vmax)
    ax[0].quiver(x, y, u, v)

    if mask and log['mask_roi'][0].shape:
        mask = log['mask_roi'][0] 
        mask = np.ma.masked_where(mask == 255, mask)
        ax[0].imshow(mask, cmap='jet', extent=[x.min(), x.max(), y.min(), y.max()])

    divider1 = make_axes_locatable(ax[0])
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)

    im2 = ax[1].imshow(cell_roi, cmap='gray',vmax=np.max(cell_roi))

    cbar = fig.colorbar(im1, cax=cax1)
    cbar.set_label('Traction stress [Pa]', rotation=270, labelpad=20)

    ax[0].set_axis_off()
    ax[1].set_axis_off()
    plt.tight_layout()
    return fig, ax
    

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

