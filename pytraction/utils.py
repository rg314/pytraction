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





    # this.nx = (int)Math.floor(((this.width - k * 2 - paramInt1) / paramInt2)) + 1;
    # this.ny = (int)Math.floor(((this.height - k * 2 - paramInt1) / paramInt2)) + 1;


#       double[][] normalizedMedianTest(double[][] paramArrayOfdouble, double paramDouble1, double paramDouble2) {
#     byte b1 = 15;
#     for (byte b2 = 0; b2 < paramArrayOfdouble.length; b2++) {
#       for (byte b = 2; b < 4; b++) {
#         double[] arrayOfDouble = getNeighbours(paramArrayOfdouble, b2, b, b1);
#         if (arrayOfDouble != null) {
#           double d = Math.abs(paramArrayOfdouble[b2][b] - getMedian(arrayOfDouble)) / (getMedian(getResidualsOfMedian(arrayOfDouble)) + paramDouble1);
#           if (d > paramDouble2)
#             paramArrayOfdouble[b2][b1] = -1.0D; 
#         } else {
#           paramArrayOfdouble[b2][b1] = -1.0D;
#         } 
#       } 
#     } 
#     return paramArrayOfdouble;
#   }
