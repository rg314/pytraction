import numpy as np

from pytraction.reg_fourier import reg_fourier_tfm
from pytraction.fourier import fourier_xu
from pytraction.optimal_lambda import optimal_lambda

class PyTraction(object):

    def __init__(self, meshsize, s, pix_per_mu, E):
        self.meshsize = meshsize
        self.s = s
        self.pix_per_mu = pix_per_mu
        self.E = E

    def calculate_traction_map(self, pos, vec, beta):         
            
        # fourier space
        grid_mat, i_max, j_max, X, fuu, Ftux, Ftuy, u = fourier_xu(pos,vec, self.meshsize, 1, self.s,[])

        # get lambda from baysian bad boi 
        L, evidencep, evidence_one = optimal_lambda(beta, fuu, Ftux, Ftuy, 1, self.s, self.meshsize, i_max, j_max, X, 1)

        # do the TFM with bays lambda
        pos,traction,traction_magnitude,f_n_m,_,_ = reg_fourier_tfm(Ftux, Ftuy, L, 1, self.s, self.meshsize, i_max, j_max, grid_mat, self.pix_per_mu, 0)

        #rescale traction with proper Young's modulus
        traction = self.E*traction
        traction_magnitude = self.E*traction_magnitude
        f_n_m = self.E*f_n_m


        # off with the shapes flip back into positon
        traction_magnitude = traction_magnitude.reshape(i_max, j_max).T
        traction_magnitude = np.flip(traction_magnitude, axis=0)

        return traction_magnitude, f_n_m, L
