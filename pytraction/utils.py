import os
import sys
from typing import Tuple, Type

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sparse
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata
from scipy.sparse import linalg as splinalg

from pytraction.dataset import Dataset


def align_slice(img: np.ndarray, ref: np.ndarray) -> Tuple[int, int, np.ndarray]:
    """Given a bead image and a ref image compute the drift using cv2.matchTemplate
    and return the drift corrected image along with the x drift (dx) and y drift (dy).
    The dx, dy shift is a measure of how much the image has moved with respect to the reference.


    Args:
        img (np.ndarray): input bead image
        ref (np.ndarray): reference bead image

    Returns:
        Tuple[int, int, np.ndarray]: dx, dy, and aligned image slice (2, w, h)
    """
    # amount to reduce template
    depth = int(min(img.shape) * 0.1)

    # calculate matchTemplate using ccorr_normed method
    tm_ccorr_normed = cv2.matchTemplate(
        img, ref[depth:-depth, depth:-depth], cv2.TM_CCORR_NORMED
    )
    max_ccorr = np.unravel_index(
        np.argmax(tm_ccorr_normed, axis=None), tm_ccorr_normed.shape
    )

    # shifts in the x and y
    dy = depth - max_ccorr[0]
    dx = depth - max_ccorr[1]

    # transformation matrix
    rows, cols = img.shape
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return dx, dy, cv2.warpAffine(img, M, (cols, rows))


def sparse_cholesky(A):
    """[summary]

    # The input matrix A must be a sparse symmetric positive-definite.

    Args:
        A ([type]): [description]

    Returns:
        [type]: [description]
    """
    n = A.shape[0]
    LU = splinalg.splu(A.tocsc(), diag_pivot_thresh=0)  # sparse LU decomposition

    return LU.L.dot(sparse.diags(LU.U.diagonal() ** 0.5)).tocsr()


def interp_vec2grid(
    pos: np.ndarray, vec: np.ndarray, cluster_size: int, grid_mat=np.array([])
) -> Tuple[np.ndarray, int, int]:
    """[summary]

    Args:
        pos (np.ndarray): [description]
        vec (np.ndarray): [description]
        cluster_size (int): [description]
        grid_mat ([type], optional): [description]. Defaults to np.array([]).

    Returns:
        Tuple[np.ndarray, int, int]: [description]
    """
    if not grid_mat:
        max_eck = [np.max(pos[0]), np.max(pos[1])]
        min_eck = [np.min(pos[0]), np.min(pos[1])]

        i_max = np.floor((max_eck[0] - min_eck[0]) / cluster_size)
        j_max = np.floor((max_eck[1] - min_eck[1]) / cluster_size)

        i_max = i_max - np.mod(i_max, 2)
        j_max = j_max - np.mod(j_max, 2)

        X = min_eck[0] + np.arange(0.5, i_max) * cluster_size
        Y = min_eck[1] + np.arange(0.5, j_max) * cluster_size

        x, y = np.meshgrid(X, Y)

        grid_mat = np.stack([x, y], axis=2)

        u = griddata(pos.T, vec.T, (x, y), method="cubic")

        return grid_mat, u, int(i_max), int(j_max)


def normalize(x: np.ndarray) -> np.ndarray:
    """[summary]

    Args:
        x (np.ndarray): [description]

    Returns:
        np.ndarray: [description]
    """
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return np.array(x * 255, dtype="uint8")


def clahe(data: np.ndarray) -> np.ndarray:
    """[summary]

    Args:
        data (np.ndarray): [description]

    Returns:
        np.ndarray: [description]
    """
    img = cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)[:, :, 0]


def bead_density(img: np.ndarray) -> float:
    """[summary]

    Args:
        img (np.ndarray): [description]

    Returns:
        float: [description]
    """

    # clahe_img = clahe(normalize(img))
    # print(clahe_img)
    # _, norm = cv2.threshold(clahe_img, 127/4, 255, cv2.THRESH_BINARY)
    # print(norm)
    # cv2.imwrite('thresh.png', norm)

    clahe_img = clahe(normalize(img))
    norm = (
        cv2.adaptiveThreshold(
            clahe_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2
        )
        / 255
    )

    ones = len(norm[norm == 1])

    area = img.shape[0] * img.shape[1]
    area_beads = ones / area

    return area_beads


def remove_boarder_from_aligned(aligned_img, aligned_ref):
    _, thresh = cv2.threshold(aligned_img, 0, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = sorted(contours, key=lambda x: cv2.contourArea(x))[0]
    x, y, w, h = cv2.boundingRect(cnt)
    return aligned_img[y : y + h, x : x + w], aligned_ref[y : y + h, x : x + w]


def plot(
    log: Type[Dataset],
    frame: int = 0,
    vmax: float = None,
    mask: bool = True,
    figsize: tuple = (16, 16),
) -> Tuple[mpl.figure.Figure, mpl.axes.Axes]:
    """[summary]

    Args:
        log (Type[Dataset]): [description]
        frame (int, optional): [description]. Defaults to 0.
        vmax (float, optional): [description]. Defaults to None.
        mask (bool, optional): [description]. Defaults to True.
        figsize (tuple, optional): [description]. Defaults to (16,16).

    Returns:
        Tuple[plt.figure.Figure,plt.axes.Axes]: [description]
    """

    log = log[frame]
    traction_map = log["traction_map"][0]
    cell_roi = log["cell_roi"][0]
    x, y = log["pos"][0]
    u, v = log["vec"][0]
    L = log["L"][0]
    vmax = np.max(traction_map) if not vmax else vmax

    fig, ax = plt.subplots(1, 2, figsize=figsize)
    im1 = ax[0].imshow(
        traction_map,
        interpolation="bicubic",
        cmap="jet",
        extent=[x.min(), x.max(), y.min(), y.max()],
        vmin=0,
        vmax=vmax,
    )
    ax[0].quiver(x, y, u, v)

    if mask and log["mask_roi"][0].shape:
        mask = log["mask_roi"][0]
        mask = np.ma.masked_where(mask == 255, mask)
        ax[0].imshow(mask, cmap="jet", extent=[x.min(), x.max(), y.min(), y.max()])

    divider1 = make_axes_locatable(ax[0])
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)

    im2 = ax[1].imshow(cell_roi, cmap="gray", vmax=np.max(cell_roi))

    cbar = fig.colorbar(im1, cax=cax1)
    cbar.set_label("Traction stress [Pa]", rotation=270, labelpad=20, size=20)
    cbar.ax.tick_params(labelsize=20)

    ax[0].set_axis_off()
    ax[1].set_axis_off()
    plt.tight_layout()
    return ax
