import numpy as np
from openpiv import widim


from pytraction.utils import allign_slice
from pytraction.fourier import fourier_xu, reg_fourier_tfm
from pytraction.optimal_lambda import optimal_lambda

    
def iterative_piv(img, ref, config):
    """
    DOCSTRING TO-DO
    """
    # allign stacks
    img = allign_slice(img, ref)

    # return aligned stack
    stack = np.stack([img, ref])

    # compute iterative PIV using openpiv
    x,y,u,v, mask = widim.WiDIM(ref.astype(np.int32), 
                                img.astype(np.int32), 
                                np.ones_like(ref).astype(np.int32),
                                **config.config['piv'])
    
    return x,y,u,v, stack

def calculate_traction_map(pos, vec, beta, meshsize, s, pix_per_mu, E):         
        
    # fourier space
    grid_mat, i_max, j_max, X, fuu, Ftux, Ftuy, u = fourier_xu(pos,vec, meshsize, 1, s,[])

    # get lambda from baysian bad boi 
    L, evidencep, evidence_one = optimal_lambda(beta, fuu, Ftux, Ftuy, 1, s, meshsize, i_max, j_max, X, 1)

    # do the TFM with bays lambda
    pos,traction,traction_magnitude,f_n_m,_,_ = reg_fourier_tfm(Ftux, Ftuy, L, 1, s, meshsize, i_max, j_max, grid_mat, pix_per_mu, 0)

    #rescale traction with proper Young's modulus
    traction = E*traction
    traction_magnitude = E*traction_magnitude
    f_n_m = E*f_n_m


    # off with the shapes flip back into positon
    traction_magnitude = traction_magnitude.reshape(i_max, j_max).T
    traction_magnitude = np.flip(traction_magnitude, axis=0)

    return traction_magnitude, f_n_m, L
