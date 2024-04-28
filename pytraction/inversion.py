import numpy as np
from scipy.fft import fftfreq, ifft2, fft2
from scipy.signal import convolve
from pytraction.kernels import *
from pytraction.utils import *


def traction_fourier(pos, vec, s, elastic_modulus, lambd=None, scaling_z=None, zdepth=0, slim=False):
    kxx, kyy, ft_ux, ft_uy, meshsize_x, meshsize_y = ft_2Dvector_field(pos=pos, vec=vec)

    if lambd is None:
        ft_gxx, ft_gxy, ft_gyy = kernel_ft(kxx, kyy, s, elastic_modulus)

    else:
        if slim:
            ft_gxx, ft_gxy, ft_gyy = kernel_ft_slim(kxx, kyy, lambd, s, elastic_modulus)

        else:
            # Get number of pixels in z-direction
            z = zdepth / scaling_z

            ft_gxx, ft_gxy, ft_gyy = kernel_ft_reg(kxx, kyy, lambd, s, elastic_modulus, z, meshsize_x, meshsize_y)

    # Calculate convolution of displacement field and Green's function in Fourier space
    ft_fx = ft_gxx * ft_ux + ft_gxy * ft_uy
    ft_fy = ft_gxy * ft_ux + ft_gyy * ft_uy

    # Avoid non-zero net force induced by spurious traction field
    ft_fx[0, 0] = 0
    ft_fy[0, 0] = 0

    # Compute inverse discrete Fourier transform of traction field
    fx = (ifft2(ft_fx)).real
    fy = (ifft2(ft_fy)).real

    # Calculate differential operator in matrix form
    gamma_glob = diff_operator(ft_gxx, ft_gxy, ft_gyy)

    return fx, fy, ft_fx, ft_fy, ft_ux, ft_uy, kxx, kyy, gamma_glob


def traction_bem(pos, method, s, elastic_modulus):
    xx = pos[:, :, 0]
    yy = pos[:, :, 1]

    i_max = xx.shape[0]
    j_max = yy.shape[1]

    x_val, y_val = xx[0, :], yy[:, 0]
    meshsize_x, meshsize_y = x_val[1] - x_val[0], y_val[1] - y_val[0]

    k_x, k_y = fftfreq(x_val.shape[0], d=meshsize_x), fftfreq(y_val.shape[0], d=meshsize_y)
    kxx, kyy = np.meshgrid(k_x, k_y)

    # Multiplication in fourier space equals convolution in spatial domain
    gamma_kxx, gamma_kxy, gamma_kyy = (kernel_ft(kxx, kyy, s, elastic_modulus) *
                                       pyramid2dim_ft(kxx, kyy, 2 * meshsize_x, 2 * meshsize_y))

    gamma_glob_xx, gamma_glob_xy, gamma_glob_yy = (np.zeros((i_max ** 2, j_max ** 2)),
                                                   np.zeros((i_max ** 2, j_max ** 2)),
                                                   np.zeros((i_max ** 2, j_max ** 2)))

    # Loop over grid in spatial domain and calculate BEM matrix components
    for i, x_k in enumerate(x_val):
        for j, y_k in enumerate(y_val):
            if method == 'conv':
                kernel_xx, kernel_xy, kernel_yy = kernel_real(xx, yy, s, elastic_modulus)
                pyr_shape = pyramid2dim(xx - x_k, yy - y_k, 2 * meshsize_x, 2 * meshsize_y)
                gamma_xx = convolve(kernel_xx,
                                    pyr_shape,
                                    mode='same')
                gamma_xy = convolve(kernel_xy,
                                    pyr_shape,
                                    mode='same')
                gamma_yy = convolve(kernel_yy,
                                    pyr_shape,
                                    mode='same')
            elif method == 'fft':
                # Use shift theorem for each node to center pyramid shape above
                shift = kxx * x_k + kyy * y_k
                gamma_xx = ifft2(gamma_kxx * np.exp(-1j * shift))
                gamma_xy = ifft2(gamma_kxy * np.exp(-1j * shift))
                gamma_yy = ifft2(gamma_kyy * np.exp(-1j * shift))

            else:
                raise ValueError(f"{method} method is not available.")

            # Define BEM matrix block sizes
            block_i = i_max * i
            block_j = j_max * j

            # Swapping rows and columns in matrix equals multiplication by permutation matrix (Involutory matrix)
            # This justifies to separate vector components as the matrices inverse is invariant under this operation
            gamma_glob_xx[block_i:block_i + i_max:, block_j:block_j + j_max] = gamma_xx
            gamma_glob_xy[block_i:block_i + i_max:, block_j:block_j + j_max] = gamma_xy
            gamma_glob_yy[block_i:block_i + i_max:, block_j:block_j + j_max] = gamma_yy
            print(f'Index {i, j}')

    # Concatenate along the columns (axis=1)
    gamma_r1 = np.concatenate((gamma_glob_xx, gamma_glob_xy), axis=1)
    gamma_r2 = np.concatenate((gamma_glob_xy, gamma_glob_yy), axis=1)

    # Concatenate along the rows (axis=0)
    gamma_glob = np.concatenate((gamma_r1, gamma_r2), axis=0)

    return gamma_glob
