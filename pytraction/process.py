import numpy as np
import cv2
from typing import Tuple
from openpiv import pyprocess, validation, filters, scaling, tools
from openpiv.windef import *
from scipy.interpolate import griddata
from pytraction.regularization import optimal_lambda_fourier
from pytraction.utils import remove_boarder_from_aligned
from pytraction.piv import extended_area_piv, widim_piv
from pytraction.regularization import *
from pytraction.inversion import traction_fourier, traction_bem


def compute_piv(img: np.ndarray, ref: np.ndarray, config):
    """
    Perform PIV on drift corrected images and returns drift corrections (dx,dy), displacement vectors (u,v) for
    positions (x,y) and image stack.
    """
    # Check if displacements are too small, which would result in an iterative decrease in window size
    if np.allclose(img, ref, rtol=1e-05, atol=1e-08):
        msg = "Image and reference frame are approximately equal (Tolerance: 1e-05)."
        raise RuntimeError(msg)

    # Calculate drift in x,y and return drift corrected img
    dx, dy, img = align_slice(img, ref)
    if config.config["settings"]["crop_aligned_slice"]:
        # Crop images to remove black borders
        img, ref = remove_boarder_from_aligned(img, ref)

    # Create drift corrected stack
    drift_corrected_stack = np.stack([img, ref])

    # Calculate displacement field
    if config.config["piv"]["piv_method"] == "widim":
        x, y, u, v = widim_piv(img, ref, config)
    elif config.config["piv"]["piv_method"] == "extended_area":
        x, y, u, v = extended_area_piv(img, ref, config)
    else:
        msg = (f'{config.config["piv"]["piv_method"]} is not a valid PIV method, please choose "widim" or'
               f'"extended_area".')
        raise RuntimeError(msg)

    # Scale field form pixels to microns
    x, y, u, v = scaling.uniform(x, y, u, v, scaling_factor=config.config["tfm"]["scaling_factor"])

    return x, y, u, v, dx, dy, drift_corrected_stack


def align_slice(img: np.ndarray, ref: np.ndarray) -> Tuple[int, int, np.ndarray]:
    """
    Given a bead and reference image, compute the drift using cv2.matchTemplate and return the drift corrected bead
    image along with the x drift (dx) and y drift (dy). The dx, dy shift is a measure of how much the image has moved
    with respect to the reference frame.
    """
    # Set window size in reference frame used for drift correction
    depth = int(min(img.shape) * 0.1)

    # Slide reference image (w x h) over input image (W x H) and calculate correlation landscape (W-w+1, H-h+1)
    # Assumes no displacements to be present in top left corner
    ccorr_normed = cv2.matchTemplate(
        img, ref[depth:-depth, depth:-depth], cv2.TM_CCORR_NORMED
    )

    # Flatten array and find index of maximal correlation
    ccorr_max_id = np.argmax(ccorr_normed, axis=None)

    # Convert flattened index back into a tuple of coordinates, describing top left corner of target rectangle
    max_ccorr = np.unravel_index(
        ccorr_max_id, ccorr_normed.shape
    )

    # Transform location of best match from correlation landscape to coordinate system of reference image
    # CC landscape is depth smaller in both x and y dimension
    dy = depth - max_ccorr[0]
    dx = depth - max_ccorr[1]
    rows, cols = img.shape

    # Initialize 2x3 transformation matrix
    # [[a, b, tx],
    # [c, d, ty]] -> a,d = 1: no scaling in x,y ; b,c = 0: no shearing or rotation ; tx, ty: translations
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])

    # Apply transformation to input image
    img = cv2.warpAffine(img, matrix, (cols, rows))

    return dx, dy, img


def interp_vec2grid(
        pos_flat: np.ndarray,
        vec_flat: np.ndarray,
        meshsize: float,
        pos_interp=np.array([])
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolates vector field onto rectangular grid which will be constructed using the meshsize parameter or can be
    handed manually using the grid_mat variable.
    """
    if len(pos_interp) == 0:
        # Get boundary dimensions of vector field to calculate grid dimensions
        max_pos = [np.max(pos_flat[0]), np.max(pos_flat[1])]  # Calculate maximum value of position along x- and y-axis
        min_pos = [np.min(pos_flat[0]), np.min(pos_flat[1])]  # Calculate minimum value of position along x- and y-axis

        # Calculate size of vector field in each direction and divide by meshsize yielding the number of mesh intervals
        i_max = np.floor((max_pos[0] - min_pos[0]) / meshsize)  # np.floor rounds result to integer number
        j_max = np.floor((max_pos[1] - min_pos[1]) / meshsize)

        # Ensure that number of mesh intervals are even by subtracting the remainder when divided by 2
        i_max = i_max - np.mod(i_max, 2)
        j_max = j_max - np.mod(j_max, 2)

        # Generate evenly spaced grid points within the calculated dimensions of the grid by adding multiples of
        # meshsize to min x-y-values
        x_grid = min_pos[0] + np.arange(0.5, i_max, 1) * meshsize
        y_grid = min_pos[1] + np.arange(0.5, j_max, 1) * meshsize

        # Creates rectangular grid from every combination of provided x and y coordinates
        xx, yy = np.meshgrid(x_grid, y_grid)  # xx and yy are both 2D matrices

        # Merges 2D xx and 2D yy array along second axis
        pos_interp = np.stack([xx, yy], axis=2)

    # Perform interpolation of displacement vectors at pos onto a grid defined by grid_mat
    vec_interp = griddata(pos_flat.T, vec_flat.T, (pos_interp[:, :, 0], pos_interp[:, :, 1]), method="cubic")

    # Remove all NaN values in the displacement field
    vec_interp = np.nan_to_num(vec_interp)

    return pos_interp, vec_interp


def calculate_traction_map(pos: np.array,
                           vec_u: np.array,
                           beta: float,
                           config):
    """
    Calculates 2D traction map given the displacement vector field and the noise value beta using an FFT or FEM
    (Boundary element method) approach.
    """
    # Get settings for TFM calculation
    poisson_ratio = config.config["tfm"]["poisson_ratio"]
    scaling_z = config.config["tfm"]["scaling_z"]
    elastic_modulus = config.config["tfm"]["elastic_modulus"]

    if config.config["tfm"]["tfm_method"] == "FT":
        # Get differential operator matrix for lambda estimation
        _, _, _, _, _, _, _, _, gamma_glob = traction_fourier(pos=pos,
                                                              vec=vec_u,
                                                              s=poisson_ratio,
                                                              elastic_modulus=elastic_modulus,
                                                              lambd=None,
                                                              scaling_z=scaling_z,
                                                              zdepth=0)
        # Predict lambda for tikhonov regularization from bayesian model
        lamd, evidence_one = optimal_lambda_fourier(
            pos=pos, vec_u=vec_u, beta=beta, elastic_modulus=elastic_modulus, s=poisson_ratio, scaling_z=scaling_z,
            gamma_glob=gamma_glob
        )

        # Calculate traction field in fourier space and transform back to spatial domain
        fx, fy, _, _, _, _, _, _, _ = traction_fourier(pos=pos,
                                                       vec=vec_u,
                                                       s=poisson_ratio,
                                                       elastic_modulus=elastic_modulus,
                                                       lambd=lamd,
                                                       scaling_z=scaling_z,
                                                       zdepth=0)

    elif config.config["tfm"]["tfm_method"] == "BEM":
        gamma_glob = traction_bem(pos=pos, method='conv', s=poisson_ratio, elastic_modulus=elastic_modulus)
        lamd, evidence_one = bayesian_regularization(vec_u, beta, gamma_glob)
        vec_f = tikhonov_reg(gamma_glob=gamma_glob, vec_u=vec_u, lambd=lamd)
        fx, fy = vec_f[:, :, 0].T, vec_f[:, :, 1].T

    else:
        msg = (f'Only fourier transform "FT" and boundary element method "BEM" are currently implemented to solve \
         inverse problem but {config.config["tfm"]["tfm_method"]} was selected.')
        raise RuntimeError(msg)

    vec_f = np.stack((fx, fy), axis=2)

    return vec_f, lamd, evidence_one
