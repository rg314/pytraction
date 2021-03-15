import sys

from scipy.sparse import linalg as splinalg
import scipy.sparse as sparse

import cv2
import numpy as np


def allign_slice(img, ref):
    """
    :param img: Image slice to allign
    :param ref: Reference bead to calculate alignment

    :return: Aligned image
    """
    # amount to reduce template
    depth = 20

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
    