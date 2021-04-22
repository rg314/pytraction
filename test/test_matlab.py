import os
import pandas as pd
import numpy as np

from pytraction.process import calculate_traction_map


def test_matlab():


    meshsize = 10 # grid spacing in pix
    pix_per_mu = 1.3
    E = 10000 # Young's modulus in Pa
    s = 0.3 # Poisson's ratio

    df = pd.read_csv(f'data{os.sep}matlab_data.csv')

    un, vn, x, y, u, v = df.T.values


    noise_vec = np.array([un.flatten(), vn.flatten()])


    varnoise = np.var(noise_vec)
    beta = 1/varnoise

    pos = np.array([x.flatten(), y.flatten()])
    vec = np.array([u.flatten(), v.flatten()])


    traction_magnitude, f_n_m, L = calculate_traction_map(pos, vec, beta, meshsize, s, pix_per_mu, E)


    output = [np.array([16,  7, 22, 22, 14,  5, 17, 10,  8, 20, 33, 30, 38, 16,  4, 23, 24,
       29, 20, 12, 50, 51, 50, 36, 22, 18, 12, 14, 60, 57, 42, 32, 15, 27,
       16, 24, 32, 46, 47, 41, 17, 10, 11, 21,  6, 28, 11, 13, 16, 22]), np.array([ 5, 18, 11, 25, 16, 23, 36, 23,  6, 17, 13, 11, 22, 16, 10, 24, 24,
        6, 19,  9, 11, 26, 20, 37, 21, 40, 54,  3, 30, 13, 12, 12, 19, 38,
       38,  9, 31, 37, 14,  6, 17, 18, 17, 11,  8, 13, 17,  9, 18,  3]), np.array([17,  3, 19, 23, 19, 28, 22, 17, 11,  8, 19, 28, 25, 15, 25,  8,  6,
       18, 39, 11, 23, 22, 36, 18, 20, 25, 23, 30, 34, 23, 10,  3, 25,  1,
       42, 29, 27, 20, 28, 25, 12, 12, 28, 20, 13, 16, 21,  6, 12,  5]), np.array([24, 24, 53, 24,  4, 11,  9,  3,  8, 29, 14, 21, 32, 12, 13, 15,  8,
       24, 15, 11, 14, 14,  5, 12, 12, 20, 13, 10, 29, 21, 27, 11, 13,  6,
       15, 20, 23, 32, 13, 35, 17, 13, 18, 24, 11, 16, 10,  7, 20, 18]), np.array([23, 28, 36, 29,  5, 20, 32, 13, 31, 33, 14,  8, 22, 17, 21, 42,  5,
       25, 18, 33, 22, 21, 13, 10, 17, 21, 11, 10, 47, 23,  7, 12,  8, 16,
       16, 33, 54, 10, 24,  1, 12, 45, 15, 17, 28, 18, 18, 12,  7, 25]), np.array([ 6, 12,  2, 17, 16, 40, 19, 18,  3,  6,  7, 21, 21, 22, 24, 28,  8,
        6, 30, 11, 28, 15, 18, 15, 20, 24, 27,  9, 16, 28, 28, 13, 23,  5,
       14, 46, 21,  4, 30,  6, 18, 25,  8, 41, 13, 18, 22, 20, 15, 17]), np.array([ 13,  29,   7,   4,  24,  15,  41,  39,  24,  14,  11,  33,  25,
        21,  30,  27,  15,   8,  31,   3,   4,   4,   8,  32,  16,  36,
        37,   5,  23,  27,  17,   9,  18,  21,  47, 108, 108,  43,  24,
        17,  12,  33,  10,  46,  22,  28,  20,  20,  21,  15]), np.array([ 16,  18,   4,  20,  34,  26,   6,  23,  14,  11,  15,   4,   9,
        36,  11,  48,  16,  33,  15,   7,  31,  23,  48,  12,  17,  11,
        21,  13,  19,  11,  18,  19,  17,  32,  81, 129, 114, 111,  26,
        26,  12,  40,  28,   7,  21,  14,   8,  21,  20,  19]), np.array([  1,  21,   6,  29,  15,  11,   6,  14,  11,   9,  17,  22,  39,
        29,  76,  77,  30,  16,  11,   9,   8,  18,  23,  26,  45,   8,
        23,   8,  17,   4,  19,  30,   5,  18,  81, 103, 115, 108,  32,
        23,  10,  18,   8,  11,  21,  23,  19,  35,   8,  10]), np.array([15, 13, 14, 21, 39,  0,  4, 11, 13,  6, 17, 26, 15, 31, 62, 67, 35,
       23, 34, 10, 23, 20, 13,  9, 21, 24, 54, 57, 20, 18, 11, 23, 27, 10,
       40, 58, 72, 42,  6, 12, 18, 30, 20, 19, 22, 18, 14, 23, 16,  5]), np.array([ 31,  15,  21,   8,  16,  19,  18,  22,  21,   8,  20,   2,  14,
        53,  49,  77,  22,  20,  11,   8,   7,   3,  35,   6,  43, 106,
       186, 129,  79,  14,  21,  33,  17,  13,  20,  29,  14,  24,   1,
        30,   8,   6,   5,  18,   8,   9,  24,   9,  19,  16]), np.array([ 27,   8,  11,  16,  30,  16,  16,  29,  12,  20,  22,  11,   6,
        41,  30,  64,  25,  22,  13,  38,  34,  35,  21,  28,  61, 181,
       252, 244, 149,  59,  23,  11,  16,  16,  25,  10,  30,  11,  30,
        12,  13,  31,  34,  29,  38,  20,  10,   7,  27,  28]), np.array([  4,  25,  37,  23,  18,  14,  41,  15,  23,   4,   6,  15,   7,
        28,   7,  17,  35,  38,  26,  36,   9,  16,   9,  16,  33, 135,
       217, 253, 147,  12,   8,  41,  25,  21,   8,   7,   6,  13,   4,
         9,  12,  30,   8,   9,  17,  23,  27,  20,  28,  25]), np.array([ 15,  17,  44,  30,  23,  17,   3,  29,  39,  13,  21,   8,  60,
        84,  76,  82,   7,  24,  21,  23,  15,  10,   7,   7,  55,  96,
       145, 146,  69,  13,  14,  25,  10,   7,  11,  33,  29,   6,  23,
         6,  28,  10,  22,  15,  32,  24,  10,   0,   5,  14]), np.array([ 13,  16,  29,  20,  21,   9,  17,  33,  34,  20,  14,  38, 124,
       148, 132,  92,  22,  37,  21,  21,  18,  19,  14,  26,  52,  48,
        59,  60,  19,  30,  19,  15,   6,  20,  20,  32,  14,  16,  13,
        11,  15,  33,  22,  19,  18,  23,  21,  14,  11,  10]), np.array([ 14,   8,  15,  20,   9,  25,  26,  19,  21,  30,   9,  18, 111,
       144, 133,  83,  33,   3,  10,  27,  19,  30,  13,  15,  11,   7,
        22,  13,  30,  15,  12,  23,  13,  16,  30,  37,  15,  39,  38,
        13,   4,  23,  25,  49,  14,  42,  39,  31,  19,   5]), np.array([  9,  21,   4,  12,  11,  35,   7,  15,   3,   9,  14,  31,  18,
       106, 112,  83,  46,  34,  30,   7,  21,  40,  20,   5,  16,  16,
        35,  22,  32,  33,  45,  14,  15,   6,  17,   7,  40,  16,  46,
        26,   6,  20,   9,   7,  19,  24,  16,  22,  22,  11]), np.array([ 4,  6, 19, 10, 32, 29, 43, 20, 10, 23, 17,  2, 35,  5, 31,  8, 11,
        3, 14, 16, 11, 19, 16, 10, 17, 20, 16, 26, 16, 24, 42, 14, 20,  4,
       32,  8, 10, 18,  3, 33, 53, 27, 27, 16, 13, 22, 45, 12, 25, 17]), np.array([27, 29, 27,  2,  6,  7, 98, 78, 27, 38,  4,  3, 23, 15, 11, 18, 11,
        7, 22,  1, 16, 13, 12, 12, 44, 25, 18, 30,  3, 35, 25, 11, 29, 21,
       15, 17,  4, 21, 46, 96, 75, 67, 17, 24, 24, 24, 34, 13, 14, 29]), np.array([ 12,  24,  10,  10,   9,  24,  47,  89,  44,  39,   8,  13,  12,
         4,  21,   6,  20,  22,  31,  17,   6,  11,  13,  26,  14,   5,
        18,  17,  29,  13,  10,   6,   2,  26,  26,  26,  26,  19, 110,
       128, 135, 108,  30,  28,  11,  25,  50,  17,  31,  18]), np.array([ 13,  12,  26,  13,  11,  36,  38,  62,  52,  21,  13,  27,  24,
        33,  11,  18,  10,  14,   6,  17,  19,  28,  39,  44,  16,   6,
        11,  34,  41,  26,  11,  20,  17,  41,   7,  34,  20,  66, 136,
       161, 172, 115,  20,  26,  15,  25,  27,  22,  37,   6]), np.array([ 29,  39,   5,  10,   9,  34,  31,  26,  25,  31,  13,  20,   5,
        45,  25,  13,  31,  17,   7,   9,  23,   8,  37,  25,  40,  12,
         5,  20,  27,  18,  14,  13,  16,  22,  40,  17,  27,   8,  70,
       138, 138,  44,  14,  36,  21,  19,  15,  16,  22,  23]), np.array([33, 19, 14,  3, 24, 27, 37,  9, 19, 18, 31, 11, 32, 31, 23, 29, 35,
       12, 19, 19, 19, 27, 53, 30, 93, 58, 52, 16, 18,  6, 14, 24,  9, 11,
       26, 18, 15, 43, 20, 33, 58, 20,  7, 12, 17, 12, 10, 41, 26, 44]), np.array([  6,  10,  12,  10,  10,  22,  17,  30,   8,  23,  18,  15,  15,
        49,  95,  65,  68,   8,  14,  11,  31,  10,  34,  99, 124,  99,
        76,  21,  14,  15,  10,  15,   7,   8,  17,  19,  16,  18,   8,
        24,  31,   1,  19,  31,  26,   7,   8,  15,  30,  14]), np.array([ 27,  16,  26,  19,  26,  20,  39,  22,  24,  20,   9,   8,  45,
       110, 163, 158,  96,  17,  11,  31,  11,  12,  26,  55,  93,  84,
        66,   1,  19,  15,  17,  12,  21,  15,   5,  19,  15,  12,  20,
        17,  25,  44,  17,  15,  12,  26,  22,  43,  20,  28]), np.array([ 24,  21,  15,   7,  31,  18,  38,  21,  17,  10,   6,  17,  49,
       123, 158, 167, 101,  28,   8,  12,   7,  17,  36,  40,  47,  32,
        35,  19,  25,  23,   8,  27,  22,   9,  15,  16,  23,  24,  10,
        27,  45,  30,  14,  14,  11,  21,   3,  27,   5,  15]), np.array([16, 12, 21, 13, 15,  8,  6,  6, 21,  5, 18, 14, 36, 71, 85, 98, 48,
       23, 18, 11, 12, 26, 30, 38, 13, 39, 22, 27,  6,  9, 21, 10, 13, 24,
       26, 25, 14,  9,  8, 22, 13, 11, 14,  7, 28, 13, 19, 36, 11,  9]), np.array([14, 15, 25, 16,  5, 10, 19, 12,  4, 13, 13, 23,  8, 32, 78, 28, 10,
       28, 10, 10,  7, 35, 18, 20, 18, 18, 24,  1, 21, 22, 20, 14, 16, 13,
       36, 40, 29,  9, 22, 17, 21, 12, 18, 14, 13, 33, 22,  8,  8, 17]), np.array([14,  6, 29, 33, 11, 37, 13,  6,  6, 26,  9, 10, 19, 41, 34, 15, 37,
        9, 26, 20, 27, 21, 20, 25, 10, 15, 24, 32, 15, 12,  8, 34, 21, 34,
       44, 27, 11, 10, 11, 19, 27, 26, 19, 21, 23, 10, 11, 14, 13,  7]), np.array([10, 28, 28, 19, 15, 12, 28, 14,  7, 23, 12, 32,  6, 19, 30, 10, 28,
       24, 24, 19, 11,  7, 12,  4, 15, 12, 10, 32, 15, 34, 45,  9, 33, 69,
       75, 56, 43,  8, 11, 34, 23, 25, 19, 18, 17, 14, 35, 20, 24, 22]), np.array([41, 31, 38, 33, 10, 24, 30, 12,  2, 13, 14, 19, 36, 42, 19,  8, 18,
        5, 20, 20, 11,  7, 26, 17,  5, 19, 11, 18, 19, 29, 25,  9, 28, 70,
       83, 74, 42,  4, 11, 22, 22,  7, 12, 28, 23, 27, 10, 15,  4, 26]), np.array([11,  7, 19, 29, 14,  4, 17,  7,  3, 16, 34,  9, 33, 14, 11, 24,  6,
       17, 24, 29, 14, 23, 11,  7, 13,  5, 19, 27, 37, 55, 34, 32, 13, 31,
       48, 49, 37, 10, 16, 10, 27, 10, 23,  6, 35, 34, 20, 26, 28, 38]), np.array([ 12,  12,  34,  26,  22,   4,  10,  27,  26,  22,   9,  16,  25,
        21,  17,  18,   8,   8,   4,   8,  15,  20,   6,  29,  15,   8,
        29,  40,  81, 113,  93,  76,  32,   5,   9,   8,  29,  31,  30,
        15,  18,  18,  10,  36,  15,  31,  24,  24,  26,  22]), np.array([ 26,   8,  16,  17,  20,  24,  22,  31,  13,  21,  24,  19,  13,
        17,  13,   4,  16,   7,  14,  69,  54,  18,  23,  38,  19,  12,
        20,  13, 108, 164, 142,  76,  15,   4,   9,  36,  12,  11,  22,
        20,  18,  11,   3,  50,  40,  11,  12,  19,  15,  47]), np.array([  6,  14,   4,   4,   8,  15,  23,  22,  21,  26,  11,  15,   6,
        11,   8,   3,   5,  33,  76, 100,  77,  58,  40,  28,  21,   4,
        17,  26,  92, 131, 132,  74,  26,  20,   7,   2,  10,   7,  10,
        19,  42,   2,   3,  41,  59,  47,  29,   7,  37,  44]), np.array([35, 31, 10,  8, 14, 18, 16, 18, 13, 62, 75, 50, 14,  3, 20, 14, 32,
       25, 67, 96, 99, 57, 37,  5, 10, 23,  9, 10, 59, 64, 69, 43, 12, 13,
        6, 12, 37, 13, 32, 30, 13, 28, 11, 19, 23, 35, 45, 36, 21, 16]), np.array([ 24,  39,  33,  23,  18,   7,  13,   2,  37, 125, 135, 115,  24,
        18,   7,   8,  13,  23,  57,  70,  75,  32,   8,  23,  17,   3,
        25,  16,  35,  17,  11,  19,  32,  26,  28,  37,   7,  19,  49,
        39,  92,  61,  10,  11,  12,   6,  26,  43,  27,  18]), np.array([ 16,   3,  21,   3,  25,  18,  12,  57, 140, 192, 209, 144,  25,
        20,  15,  32,  21,  25,  14,  47,  21,  28,  17,  15,  18,  19,
        17,  20,  34,  26,  24,   7,  33,  24,   4,  18,  25,  11,  62,
       109, 126, 111,  46,  13,  17,   8,  47,  34,   7,  30]), np.array([ 39,  27,  25,  22,  18,  40,  49,  53, 146, 200, 220, 148,  51,
        25,  15,  33,  29,  14,  20,  22,  10,  20,  22,  10,  11,  24,
        25,  60,  82, 121, 144,  82,  14,  26,  28,  14,   4,   8,  44,
       102, 118,  80,  54,  29,  25,  37,  15,  29,  38,  16]), np.array([  3,   9,  35,  28,   8,  16,  49,  37,  75, 123, 123,  77,  38,
        17,  13,  26,  16,  19,  19,  25,  13,  10,   8,  15,  29,  18,
        20,  44, 168, 207, 209, 157,  27,   3,  12,   0,  13,  16,  32,
        95, 100,  72,  33,  11,   6,  19,  20,  20,  28,  16]), np.array([ 25,  23,  26,   7,  25,  27,  15,  13,  40,  45,  36,   4,  32,
        17,  20,  10,  10,  10,  12,  13,  12,  12,  28,  13,  28,  12,
        13,  62, 197, 240, 239, 125,  37,  11,  28,  13,  10,   7,  15,
        43,  33,  16,  15,  19,  12,  24,  21,   2,  12,  21]), np.array([ 23,  33,  12,  10,  12,  12,  17,  31,  36,  14,  11,  12,   3,
        18,   7,  35,  48,  28,  21,  31,  32,  35,  20,  24,   9,  25,
        21,  39, 123, 184, 180,  86,  11,  15,  11,  14,  18,  16,   5,
        10,   6,  30,  17,  11,  24,  54,  16,  36,  21,  13]), np.array([ 5,  1, 14, 18, 13, 11, 16, 42, 37, 36,  8,  7, 16, 10, 12, 39, 40,
       32, 15, 20, 36, 22, 10, 29, 17, 25, 26, 10, 48, 87, 75, 19, 10, 13,
        3, 31, 15, 17, 10, 27, 14, 19, 22, 18, 37, 11, 17, 36, 10, 19]), np.array([23, 12, 21, 17, 11,  4, 15,  6, 12, 18,  8, 18, 21, 12, 25, 40, 10,
       32,  6, 15, 22, 10, 10,  8, 13, 25,  4,  7, 34, 29, 25, 13, 26, 21,
       47,  9,  7, 11,  6, 21,  1, 22, 30,  9, 20,  9, 17, 29, 37, 60]), np.array([33, 37,  4, 25, 31, 20, 16,  3, 22, 11, 16, 30, 22, 28, 59, 33,  4,
       23,  9, 15, 15, 27,  7,  4, 10, 21, 17, 12, 21, 14, 23, 21, 21,  9,
       32, 11, 13, 21,  5, 18,  9, 36, 11, 25,  3, 12, 20, 10, 35, 24]), np.array([24, 23, 10, 14, 14, 23, 14, 15,  8, 33, 18, 21,  6, 25, 32, 41,  8,
       12,  9, 12, 28,  9, 16, 15, 12, 20, 28, 11,  7,  5, 22, 48, 16, 15,
       39, 27, 23, 12, 11, 24, 14, 12,  9, 20,  8,  9,  9, 22, 33, 37]), np.array([ 8, 23, 13, 13, 13, 27, 12, 31, 10, 25,  5, 25, 11, 12,  6, 42, 15,
       18, 22, 23, 28, 30, 24,  5,  5,  9, 15, 22, 20, 11,  8, 29, 42, 24,
       30, 22,  9, 15, 30, 10,  1, 24, 18, 30, 13, 16, 25, 12, 12, 26]), np.array([17, 37, 17, 18, 19, 24, 27, 23,  3,  9, 23,  7,  7, 11, 25,  9, 11,
       19,  9, 37, 33, 12, 13, 22, 14, 17, 15, 11,  4,  5, 19, 11, 38, 21,
        8,  6,  2, 12, 31, 21, 11,  9, 24, 44, 29, 34, 19, 13, 12, 34]), np.array([25, 23, 32, 24,  7,  3, 32, 21, 17,  6, 23, 27, 21, 19, 17, 20, 36,
       40, 21, 15, 36, 18, 16, 23, 35, 22,  4, 43, 48, 31, 21, 23, 27, 22,
       25,  6,  8, 20,  5, 14, 20, 16, 20, 25,  8, 25, 23, 16,  9, 26]), np.array([10, 18, 16,  5, 34, 11,  3, 10, 11, 19, 22, 13, 15, 31, 25,  7, 45,
       20, 11, 21, 41, 46,  8, 28, 17, 11, 44, 49, 41, 21, 12, 13,  8, 19,
       28,  0,  6, 20, 13, 11,  6, 23, 14, 20, 16,  3, 13, 31, 23, 29])]


    for (a,b) in zip(traction_magnitude.astype('int'),np.array(output).astype('int')):
        for (c,d) in zip(a,b):
            assert c == d, 'output traction map does not match'

    assert int(np.real(L)) == 145, f'L estimated incorrectly {np.real(L)}'

