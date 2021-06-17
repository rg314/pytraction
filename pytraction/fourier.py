import numpy as np
from scipy.sparse import spdiags

from pytraction.interp_vec2grid import interp_vec2grid

def fourier_xu(pos, vec, meshsize, E, s, grid_mat):
    
    new_pos = pos + vec # shifted positions of data by displacements
    
    # interpolate data onto rectangular grid. If grid_mat is empty, a new grid
    # will be constructed. Otherwise the grid in grid_mat will be used.
    grid_mat,u, i_max,j_max = interp_vec2grid(new_pos, vec, meshsize, grid_mat)

    # shapes might be off here!
    # construct wave vectors
    kx_vec = 2*np.pi/i_max/meshsize*np.concatenate([np.arange(0,(i_max-1)/2), -np.arange(i_max/2,0, -1)])
    kx_vec = np.expand_dims(kx_vec, axis=0)
    ky_vec = 2*np.pi/j_max/meshsize*np.concatenate([np.arange(0,(j_max-1)/2), -np.arange(j_max/2,0, -1)])
    ky_vec = np.expand_dims(ky_vec, axis=0)

    kx = np.tile(kx_vec.T, (1,j_max))
    ky = np.tile(ky_vec, (i_max, 1))

    # We ignore DC component below and can therefore set k(1,1) =1
    kx[0,0] = 1
    ky[0,0] = 1
    k = np.sqrt(kx**2 + ky**2)
  
    # calculate Green's function
    conf = 2*(1+s)/(E*k**3)
    gf_xx = conf * ((1-s)*k**2+s*ky**2)
    gf_xy = conf * (-s*kx*ky)
    gf_yy = conf * ((1-s)*k**2+s*kx**2)

    # set DC component to one
    gf_xx[0,0] = 0
    gf_xy[0,0] = 0
    gf_yy[0,0] = 0

    # FT of the real matrix is symmetric
    gf_xy[int(i_max//2),:] = 0
    gf_xy[:, int(j_max//2)] = 0 

    # reshape stuff to produce a large sparse matrix X
    g1 = gf_xx.reshape(1,i_max*j_max)
    g2 = gf_yy.reshape(1,i_max*j_max)

    X1 = np.array([g1,g2]).T.flatten()
    X1 = np.expand_dims(X1, axis=1)

  
    g3 = gf_xy.reshape(1, i_max*j_max)
    g4 = np.zeros(g3.shape)

    X2 = np.array([g3,g4]).T.flatten()
    X2 = np.expand_dims(X2, axis=1)
    X3 = X2[1:]


    pad = np.expand_dims(np.array([0]), axis=1)
    data = np.array([np.concatenate([X3,pad]).T, X1.T, np.concatenate([pad, X3]).T])
    data = np.squeeze(data, axis=1)
    X = spdiags(data, (-1,0,1), len(X1), len(X1))


    # remove any nan values in the displacement field #iss14
    u = np.nan_to_num(u)

    # Fourier transform displacement field
    ftux = np.fft.fft2(u[:,:,0]).T
    ftuy = np.fft.fft2(u[:,:,1]).T

    fux1 = ftux.reshape(i_max*j_max,1)
    fux2 = ftuy.reshape(i_max*j_max,1)

    fuu = np.array([fux1,fux2]).T.flatten()
    fuu = np.expand_dims(fuu, axis=1)

    return grid_mat, i_max, j_max, X, fuu, ftux, ftuy, u
