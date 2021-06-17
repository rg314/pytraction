import numpy as np
from scipy.sparse import spdiags
from pytraction.utils import interp_vec2grid

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


def reg_fourier_tfm(Ftux,Ftuy,L,E,s,cluster_size,i_max, j_max, grid_mat=None, pix_durch_my=None,zdepth=None, slim=False):

    V = 2*(1+s)/E
    # shapes might be off here!
    # construct wave vectors
    kx_vec = 2*np.pi/i_max/cluster_size*np.concatenate([np.arange(0,(i_max-1)/2), -np.arange(i_max/2,0, -1)])
    kx_vec = np.expand_dims(kx_vec, axis=0)
    ky_vec = 2*np.pi/j_max/cluster_size*np.concatenate([np.arange(0,(j_max-1)/2), -np.arange(j_max/2,0, -1)])
    ky_vec = np.expand_dims(ky_vec, axis=0)

    kx = np.tile(kx_vec.T, (1,j_max))
    ky = np.tile(ky_vec, (i_max, 1))

    # We ignore DC component below and can therefore set k(1,1) =1
    kx[0,0] = 1
    ky[0,0] = 1

    if slim:   #Slim output. Calculate only traction forces for the case z=0
        Ginv_xx = (kx**2+ky**2)**(-1/2)*V*(kx**2*L+ky**2*L+V**2)**(-1)*(kx**2* \
                L+ky**2*L+((-1)+s)**2*V**2)**(-1)*(kx**4*(L+(-1)*L*s)+ \
                kx**2*((-1)*ky**2*L*((-2)+s)+(-1)*((-1)+s)*V**2)+ky**2*( \
                ky**2*L+((-1)+s)**2*V**2))
        Ginv_yy =  (kx**2+ky**2)**(-1/2)*V*(kx**2*L+ky**2*L+V**2)**(-1)*(kx**2* \
                L+ky**2*L+((-1)+s)**2*V**2)**(-1)*(kx**4*L+(-1)*ky**2*((-1)+ \
                s)*(ky**2*L+V**2)+kx**2*((-1)*ky**2*L*((-2)+s)+((-1)+s)**2* \
                V**2))
        Ginv_xy =  (-1)*kx*ky*(kx**2+ky**2)**(-1/2)*s*V*(kx**2*L+ky**2*L+ \
                V**2)**(-1)*(kx**2*L+ky**2*L+((-1)+s)*V**2)*(kx**2*L+ky**2* \
                L+((-1)+s)**2*V**2)**(-1)

        Ginv_xx[0,0] = 0
        Ginv_yy[0,0] = 0
        Ginv_xy[0,0] = 0
        
        Ginv_xy[int(i_max/2),:] = 0
        Ginv_xy[:,int(j_max/2)] = 0
        Ftfx = Ginv_xx*Ftux + Ginv_xy*Ftuy
        Ftfy = Ginv_xy*Ftux + Ginv_yy*Ftuy
        
        # simply set variables that we do not need to calculate here to 0 
        f_pos = 0
        f_nm_2 = 0
        f_magnitude = 0
        f_n_m = 0

        return f_pos, f_nm_2, f_magnitude, f_n_m, Ftfx, Ftfy
         
         
    else:   #full output, calculate traction forces with z>=0 
        z = zdepth/pix_durch_my 
        X = i_max*cluster_size/2
        Y = j_max*cluster_size/2
        if z == 0:
            g0x = np.pi**(-1)*V*((-1)*Y*np.log((-1)*X+np.sqrt(X**2+Y**2))+Y*np.log( \
                X+np.sqrt(X**2+Y**2))+((-1)+s)*X*(np.log((-1)*Y+np.sqrt(X**2+Y**2) \
                )+(-1)*np.log(Y+np.sqrt(X**2+Y**2))))

            g0y = np.pi**(-1)*V*(((-1)+s)*Y*(np.log((-1)*X+np.sqrt(X**2+Y**2))+( \
                -1)*np.log(X+np.sqrt(X**2+Y**2)))+X*((-1)*np.log((-1)*Y+np.sqrt( \
                X**2+Y**2))+np.log(Y+np.sqrt(X**2+Y**2))))

        else:
            g0x = np.pi**(-1)*V*(((-1)+2*s)*z*np.arctan(X**(-1)*Y)+(-2)*z* \
            np.arctan(X*Y*z**(-1)*(X**2+Y**2+z**2)**(-1/2))+z*np.arctan(X**( \
            -1)*Y*z*(X**2+Y**2+z**2)**(-1/2))+(-2)*s*z*np.arctan(X**( \
            -1)*Y*z*(X**2+Y**2+z**2)**(-1/2))+(-1)*Y*np.log((-1)*X+ \
            np.sqrt(X**2+Y**2+z**2))+Y*np.log(X+np.sqrt(X**2+Y**2+z**2))+(-1)* \
            X*np.log((-1)*Y+np.sqrt(X**2+Y**2+z**2))+s*X*np.log((-1)*Y+np.sqrt( \
            X**2+Y**2+z**2))+(-1)*((-1)+s)*X*np.log(Y+np.sqrt(X**2+Y**2+ \
            z**2)))

            g0y = (-1)*np.pi**(-1)*V*(((-1)+2*s)*z*np.arctan(X**(-1)*Y)+(3+(-2) \
            *s)*z*np.arctan(X*Y*z**(-1)*(X**2+Y**2+z**2)**(-1/2))+z* \
            np.arctan(X**(-1)*Y*z*(X**2+Y**2+z**2)**(-1/2))+(-2)*s*z* \
            np.arctan(X**(-1)*Y*z*(X**2+Y**2+z**2)**(-1/2))+Y*np.log((-1)* \
            X+np.sqrt(X**2+Y**2+z**2))+(-1)*s*Y*np.log((-1)*X+np.sqrt(X**2+ \
            Y**2+z**2))+((-1)+s)*Y*np.log(X+np.sqrt(X**2+Y**2+z**2))+X*np.log( \
            (-1)*Y+np.sqrt(X**2+Y**2+z**2))+(-1)*X*np.log(Y+np.sqrt(X**2+Y**2+ \
            z**2)))  

        Ginv_xx =np.exp(np.sqrt(kx**2+ky**2)*z)*(kx**2+ky**2)**(-1/2)*V*(np.exp( \
            2*np.sqrt(kx**2+ky**2)*z)*(kx**2+ky**2)*L+V**2)**(-1)*(4* \
            ((-1)+s)*V**2*((-1)+s+np.sqrt(kx**2+ky**2)*z)+(kx**2+ky**2) \
            *(4*np.exp(2*np.sqrt(kx**2+ky**2)*z)*L+V**2*z**2))**(-1)*(( \
            -2)*np.exp(2*np.sqrt(kx**2+ky**2)*z)*(kx**2+ky**2)*L*((-2)* \
            ky**2+kx**2*((-2)+2*s+np.sqrt(kx**2+ky**2)*z))+V**2*( \
            kx**2*(4+(-4)*s+(-2)*np.sqrt(kx**2+ky**2)*z+ky**2*z**2)+ \
            ky**2*(4+4*((-2)+s)*s+(-4)*np.sqrt(kx**2+ky**2)*z+4*np.sqrt( \
            kx**2+ky**2)*s*z+ky**2*z**2)))
        Ginv_yy = np.exp(np.sqrt(kx**2+ky**2)*z)*(kx**2+ky**2)**(-1/2)*V*(np.exp( \
            2*np.sqrt(kx**2+ky**2)*z)*(kx**2+ky**2)*L+V**2)**(-1)*(4* \
            ((-1)+s)*V**2*((-1)+s+np.sqrt(kx**2+ky**2)*z)+(kx**2+ky**2) \
            *(4*np.exp(2*np.sqrt(kx**2+ky**2)*z)*L+V**2*z**2))**(-1)*( \
            2*np.exp(2*np.sqrt(kx**2+ky**2)*z)*(kx**2+ky**2)*L*(2* \
            kx**2+(-1)*ky**2*((-2)+2*s+np.sqrt(kx**2+ky**2)*z))+V**2*( \
            kx**4*z**2+(-2)*ky**2*((-2)+2*s+np.sqrt(kx**2+ky**2)*z)+ \
            kx**2*(4+4*((-2)+s)*s+(-4)*np.sqrt(kx**2+ky**2)*z+4*np.sqrt( \
            kx**2+ky**2)*s*z+ky**2*z**2)))
        Ginv_xy = (-1)*np.exp(np.sqrt(kx**2+ky**2)*z)*kx*ky*(kx**2+ky**2)**( \
            -1/2)*V*(np.exp(2*np.sqrt(kx**2+ky**2)*z)*(kx**2+ky**2)*L+ \
            V**2)**(-1)*(2*np.exp(2*np.sqrt(kx**2+ky**2)*z)*(kx**2+ky**2) \
            *L*(2*s+np.sqrt(kx**2+ky**2)*z)+V**2*(4*((-1)+s)*s+(-2) \
            *np.sqrt(kx**2+ky**2)*z+4*np.sqrt(kx**2+ky**2)*s*z+(kx**2+ \
            ky**2)*z**2))*(4*((-1)+s)*V**2*((-1)+s+np.sqrt(kx**2+ \
            ky**2)*z)+(kx**2+ky**2)*(4*np.exp(2*np.sqrt(kx**2+ky**2)*z)* \
            L+V**2*z**2))**(-1)

        Ginv_xx[0,0] = 1/g0x
        Ginv_yy[0,0] = 1/g0y
        Ginv_xy[0,0] = 0

        Ginv_xy[int(i_max//2),:] = 0
        Ginv_xy[:, int(j_max//2)] = 0

        Ftfx = Ginv_xx*Ftux + Ginv_xy*Ftuy
        Ftfy = Ginv_xy*Ftux + Ginv_yy*Ftuy

        f_n_m = np.zeros(Ftfx.shape+(2,))
        f_n_m[:,:,0] =  np.real(np.fft.ifft2(Ftfx))
        f_n_m[:,:,1] =  np.real(np.fft.ifft2(Ftfy))

        f_nm_2 = np.zeros((i_max*j_max,2,1))
        f_nm_2[:,0] = f_n_m[:,:,0].reshape(i_max*j_max,1)
        f_nm_2[:,1] = f_n_m[:,:,1].reshape(i_max*j_max,1)


        f_pos = np.zeros((i_max*j_max,2,1))
        f_pos[:,0] =  grid_mat[:,:,0].reshape(i_max*j_max,1)
        f_pos[:,1] =  grid_mat[:,:,1].reshape(i_max*j_max,1)

        f_magnitude = np.sqrt(f_nm_2[:,0]**2 + f_nm_2[:,1]**2)

    return f_pos, f_nm_2, f_magnitude, f_n_m, Ftfx, Ftfy
