import numpy as np


# Define forward kernel component functions
def kernel_real(x, y, s, elastic_modul):
    coeff = (1 + s) / (np.pi * elastic_modul)
    r = np.sqrt(x ** 2 + y ** 2)
    gxx = coeff * ((1 - s) / r + s * x ** 2 / r ** 3)
    gxy = coeff * (s * x * y / r ** 3)
    gyy = coeff * ((1 - s) / r + s * y ** 2 / r ** 3)
    return np.nan_to_num(gxx), np.nan_to_num(gxy), np.nan_to_num(gyy)


# Define forward kernel component functions in fourier space
def kernel_ft(k_x, k_y, s, elastic_modul):
    coeff = 2 * (1 + s) / elastic_modul
    k = np.sqrt(k_x ** 2 + k_y ** 2)
    k[0, 0] = 1

    gxx = coeff * ((1 - s) / k + s * k_y ** 2 / k ** 3)
    gyy = coeff * ((1 - s) / k + s * k_x ** 2 / k ** 3)
    gxy = coeff * (s * k_x * k_y / k ** 3)

    # Set all zero frequency components in greens function to zero
    gxx[0, 0] = 0
    gxy[0, 0] = 0
    gyy[0, 0] = 0

    #  Set values in middle row and column to zero
    i_max = len(k_x[0, :])
    j_max = len(k_y[:, 0])

    gxy[int(i_max // 2), :] = 0
    gxy[:, int(j_max // 2)] = 0

    return gxx, gxy, gyy


def kernel_ft_slim(kxx, kyy, L, s, E):
    v = 2 * (1 + s) / E
    k = np.sqrt(kxx ** 2 + kyy ** 2)
    k[0, 0] = 1
    k_inv = k ** (-1)

    ft_gxx = (
            k_inv
            * v
            * (kxx ** 2 * L + kyy ** 2 * L + v ** 2) ** (-1)
            * (kxx ** 2 * L + kyy ** 2 * L + ((-1) + s) ** 2 * v ** 2) ** (-1)
            * (
                    kxx ** 4 * (L + (-1) * L * s)
                    + kxx ** 2
                    * ((-1) * kyy ** 2 * L * ((-2) + s) + (-1) * ((-1) + s) * v ** 2)
                    + kyy ** 2 * (kyy ** 2 * L + ((-1) + s) ** 2 * v ** 2)
            )
    )

    ft_gxy = (
            (-1)
            * kxx
            * kyy
            * k_inv
            * s
            * v
            * (kxx ** 2 * L + kyy ** 2 * L + v ** 2) ** (-1)
            * (kxx ** 2 * L + kyy ** 2 * L + ((-1) + s) * v ** 2)
            * (kxx ** 2 * L + kyy ** 2 * L + ((-1) + s) ** 2 * v ** 2) ** (-1)
    )

    ft_gyy = (
            k_inv
            * v
            * (kxx ** 2 * L + kyy ** 2 * L + v ** 2) ** (-1)
            * (kxx ** 2 * L + kyy ** 2 * L + ((-1) + s) ** 2 * v ** 2) ** (-1)
            * (
                    kxx ** 4 * L
                    + (-1) * kyy ** 2 * ((-1) + s) * (kyy ** 2 * L + v ** 2)
                    + kxx ** 2 * ((-1) * kyy ** 2 * L * ((-2) + s) + ((-1) + s) ** 2 * v ** 2)
            )
    )

    # Set all zero frequency components in Green's function to zero
    ft_gxx[0, 0] = 0
    ft_gxy[0, 0] = 0
    ft_gyy[0, 0] = 0

    i_max = (np.shape(kxx)[0])
    j_max = (np.shape(kyy)[1])

    ft_gxy[int(i_max / 2), :] = 0
    ft_gxy[:, int(j_max / 2)] = 0

    return ft_gxx, ft_gxy, ft_gyy


def kernel_ft_reg(kxx, kyy, L, s, elastic_modulus, z, meshsize_x, meshsize_y):
    i_max = (np.shape(kxx)[0])
    j_max = (np.shape(kyy)[1])

    v = 2 * (1 + s) / elastic_modulus
    k = np.sqrt(kxx ** 2 + kyy ** 2)
    k[0, 0] = 1
    k_inv = k ** (-1)

    # Calculate center coordinates of x- and y-axis
    X = i_max * meshsize_x / 2
    Y = j_max * meshsize_y / 2

    if z == 0:
        # Derive normalization factors for the forward Green's function matrix
        g0x = (
                np.pi ** (-1)
                * v
                * (
                        (-1) * Y * np.log((-1) * X + np.sqrt(X ** 2 + Y ** 2))
                        + Y * np.log(X + np.sqrt(X ** 2 + Y ** 2))
                        + ((-1) + s)
                        * X
                        * (
                                np.log((-1) * Y + np.sqrt(X ** 2 + Y ** 2))
                                + (-1) * np.log(Y + np.sqrt(X ** 2 + Y ** 2))
                        )
                )
        )

        g0y = (
                np.pi ** (-1)
                * v
                * (
                        ((-1) + s)
                        * Y
                        * (
                                np.log((-1) * X + np.sqrt(X ** 2 + Y ** 2))
                                + (-1) * np.log(X + np.sqrt(X ** 2 + Y ** 2))
                        )
                        + X
                        * (
                                (-1) * np.log((-1) * Y + np.sqrt(X ** 2 + Y ** 2))
                                + np.log(Y + np.sqrt(X ** 2 + Y ** 2))
                        )
                )
        )

    else:
        # Derive normalization factors for the forward Green's function matrix
        g0x = (
                np.pi ** (-1)
                * v
                * (
                        ((-1) + 2 * s) * z * np.arctan(X ** (-1) * Y)
                        + (-2)
                        * z
                        * np.arctan(
                    X * Y * z ** (-1) * (X ** 2 + Y ** 2 + z ** 2) ** (-1 / 2)
                )
                        + z
                        * np.arctan(
                    X ** (-1) * Y * z * (X ** 2 + Y ** 2 + z ** 2) ** (-1 / 2)
                )
                        + (-2)
                        * s
                        * z
                        * np.arctan(
                    X ** (-1) * Y * z * (X ** 2 + Y ** 2 + z ** 2) ** (-1 / 2)
                )
                        + (-1) * Y * np.log((-1) * X + np.sqrt(X ** 2 + Y ** 2 + z ** 2))
                        + Y * np.log(X + np.sqrt(X ** 2 + Y ** 2 + z ** 2))
                        + (-1) * X * np.log((-1) * Y + np.sqrt(X ** 2 + Y ** 2 + z ** 2))
                        + s * X * np.log((-1) * Y + np.sqrt(X ** 2 + Y ** 2 + z ** 2))
                        + (-1)
                        * ((-1) + s)
                        * X
                        * np.log(Y + np.sqrt(X ** 2 + Y ** 2 + z ** 2))
                )
        )

        g0y = (
                (-1)
                * np.pi ** (-1)
                * v
                * (
                        ((-1) + 2 * s) * z * np.arctan(X ** (-1) * Y)
                        + (3 + (-2) * s)
                        * z
                        * np.arctan(
                    X * Y * z ** (-1) * (X ** 2 + Y ** 2 + z ** 2) ** (-1 / 2)
                )
                        + z
                        * np.arctan(
                    X ** (-1) * Y * z * (X ** 2 + Y ** 2 + z ** 2) ** (-1 / 2)
                )
                        + (-2)
                        * s
                        * z
                        * np.arctan(
                    X ** (-1) * Y * z * (X ** 2 + Y ** 2 + z ** 2) ** (-1 / 2)
                )
                        + Y * np.log((-1) * X + np.sqrt(X ** 2 + Y ** 2 + z ** 2))
                        + (-1)
                        * s
                        * Y
                        * np.log((-1) * X + np.sqrt(X ** 2 + Y ** 2 + z ** 2))
                        + ((-1) + s) * Y * np.log(X + np.sqrt(X ** 2 + Y ** 2 + z ** 2))
                        + X * np.log((-1) * Y + np.sqrt(X ** 2 + Y ** 2 + z ** 2))
                        + (-1) * X * np.log(Y + np.sqrt(X ** 2 + Y ** 2 + z ** 2))
                )
        )

    # Derive components of the Green's function matrix
    ft_gxx = (
            np.exp(np.sqrt(kxx ** 2 + kyy ** 2) * z)
            * k_inv
            * v
            * (
                    np.exp(2 * np.sqrt(kxx ** 2 + kyy ** 2) * z) * (kxx ** 2 + kyy ** 2) * L
                    + v ** 2
            )
            ** (-1)
            * (
                    4 * ((-1) + s) * v ** 2 * ((-1) + s + np.sqrt(kxx ** 2 + kyy ** 2) * z)
                    + (kxx ** 2 + kyy ** 2)
                    * (4 * np.exp(2 * np.sqrt(kxx ** 2 + kyy ** 2) * z) * L + v ** 2 * z ** 2)
            )
            ** (-1)
            * (
                    (-2)
                    * np.exp(2 * np.sqrt(kxx ** 2 + kyy ** 2) * z)
                    * (kxx ** 2 + kyy ** 2)
                    * L
                    * (
                            (-2) * kyy ** 2
                            + kxx ** 2 * ((-2) + 2 * s + np.sqrt(kxx ** 2 + kyy ** 2) * z)
                    )
                    + v ** 2
                    * (
                            kxx ** 2
                            * (
                                    4
                                    + (-4) * s
                                    + (-2) * np.sqrt(kxx ** 2 + kyy ** 2) * z
                                    + kyy ** 2 * z ** 2
                            )
                            + kyy ** 2
                            * (
                                    4
                                    + 4 * ((-2) + s) * s
                                    + (-4) * np.sqrt(kxx ** 2 + kyy ** 2) * z
                                    + 4 * np.sqrt(kxx ** 2 + kyy ** 2) * s * z
                                    + kyy ** 2 * z ** 2
                            )
                    )
            )
    )

    ft_gyy = (
            np.exp(np.sqrt(kxx ** 2 + kyy ** 2) * z)
            * k_inv
            * v
            * (
                    np.exp(2 * np.sqrt(kxx ** 2 + kyy ** 2) * z) * (kxx ** 2 + kyy ** 2) * L
                    + v ** 2
            )
            ** (-1)
            * (
                    4 * ((-1) + s) * v ** 2 * ((-1) + s + np.sqrt(kxx ** 2 + kyy ** 2) * z)
                    + (kxx ** 2 + kyy ** 2)
                    * (4 * np.exp(2 * np.sqrt(kxx ** 2 + kyy ** 2) * z) * L + v ** 2 * z ** 2)
            )
            ** (-1)
            * (
                    2
                    * np.exp(2 * np.sqrt(kxx ** 2 + kyy ** 2) * z)
                    * (kxx ** 2 + kyy ** 2)
                    * L
                    * (
                            2 * kxx ** 2
                            + (-1) * kyy ** 2 * ((-2) + 2 * s + np.sqrt(kxx ** 2 + kyy ** 2) * z)
                    )
                    + v ** 2
                    * (
                            kxx ** 4 * z ** 2
                            + (-2) * kyy ** 2 * ((-2) + 2 * s + np.sqrt(kxx ** 2 + kyy ** 2) * z)
                            + kxx ** 2
                            * (
                                    4
                                    + 4 * ((-2) + s) * s
                                    + (-4) * np.sqrt(kxx ** 2 + kyy ** 2) * z
                                    + 4 * np.sqrt(kxx ** 2 + kyy ** 2) * s * z
                                    + kyy ** 2 * z ** 2
                            )
                    )
            )
    )

    ft_gxy = (
            (-1)
            * np.exp(np.sqrt(kxx ** 2 + kyy ** 2) * z)
            * kxx
            * kyy
            * k_inv
            * v
            * (
                    np.exp(2 * np.sqrt(kxx ** 2 + kyy ** 2) * z) * (kxx ** 2 + kyy ** 2) * L
                    + v ** 2
            )
            ** (-1)
            * (
                    2
                    * np.exp(2 * np.sqrt(kxx ** 2 + kyy ** 2) * z)
                    * (kxx ** 2 + kyy ** 2)
                    * L
                    * (2 * s + np.sqrt(kxx ** 2 + kyy ** 2) * z)
                    + v ** 2
                    * (
                            4 * ((-1) + s) * s
                            + (-2) * np.sqrt(kxx ** 2 + kyy ** 2) * z
                            + 4 * np.sqrt(kxx ** 2 + kyy ** 2) * s * z
                            + (kxx ** 2 + kyy ** 2) * z ** 2
                    )
            )
            * (
                    4 * ((-1) + s) * v ** 2 * ((-1) + s + np.sqrt(kxx ** 2 + kyy ** 2) * z)
                    + (kxx ** 2 + kyy ** 2)
                    * (4 * np.exp(2 * np.sqrt(kxx ** 2 + kyy ** 2) * z) * L + v ** 2 * z ** 2)
            )
            ** (-1)
    )

    ft_gxx[0, 0] = 1 / g0x
    ft_gxy[0, 0] = 0
    ft_gyy[0, 0] = 1 / g0y

    # ToDo: Check if this is the correct approach (Set Ginv_xy for lowest frequencies to zero)
    ft_gxy[int(i_max // 2), :] = 0
    ft_gxy[:, int(j_max // 2)] = 0

    return ft_gxx, ft_gxy, ft_gyy
