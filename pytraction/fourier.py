from pytraction.interp_vec2grid import interp_vec2grid
import numpy as np

def fourier_xu(pos, vec, meshsize, E, s, grid_mat):
    
    new_pos = pos + vec # shifted positions of data by displacements
    
    # interpolate data onto rectangular grid. If grid_mat is empty, a new grid
    # will be constructed. Otherwise the grid in grid_mat will be used.
    grid_mat,u, i_max,j_max = interp_vec2grid(new_pos, vec, meshsize, grid_mat)

    # shapes might be off here!
#   %construct wave vectors
    kx_vec = 2*np.pi/i_max/meshsize*np.concatenate([np.arange(0,(i_max-1)/2), -np.arange(i_max/2,0, -1)])
    kx_vec = np.expand_dims(kx_vec, axis=0)
    ky_vec = 2*np.pi/j_max/meshsize*np.concatenate([np.arange(0,(j_max-1)/2), -np.arange(j_max/2,0, -1)])
    ky_vec = np.expand_dims(ky_vec, axis=0)

    kx = np.tile(kx_vec.T, (1,j_max))
    ky = np.tile(ky_vec, (i_max, 1))

#   %We ignore DC component below and can therefore set k(1,1) =1
    kx[0,0] = 1
    ky[0,0] = 1
    k = np.sqrt(kx**2 + ky**2)
  
#   %calculate Green's function
    conf = 2*(1+s)/(E*k**3)
    gf_xx = conf * ((1-s)*k**2+s*ky**2)
    gf_xy = conf * (-s*kx*ky)
    gf_yy = conf * ((1-s)*k**2+s*kx**2)

#   %set DC component to one
    gf_xx[0,0] = 0
    gf_xy[0,0] = 0
    gf_yy[0,0] = 0

#   %FT of the real matrix is symmetric
    gf_xy = 
    gf_xy = 

#   Gf_xy(i_max/2+1,:) = 0;
#   Gf_xy(:,j_max/2+1) = 0;
  
#   %reshape stuff to produce a large sparse matrix X
#   G1 = reshape(Gf_xx,[1,i_max*j_max]);
#   G2 = reshape(Gf_yy,[1,i_max*j_max]);
#   X1 = reshape([G1; G2], [], 1)';
  
#   G3 = reshape(Gf_xy,[1,i_max*j_max]);
#   G4 = zeros(1, i_max*j_max); 
#   X2 = reshape([G4; G3], [], 1)';
#   X3 = X2(1,2:end);
   
#   X = spdiags([[X3 0]' X1' [0 X3]'],-1:1,length(X1),length(X1));
  
#   %Fourier transform displacement field
#   Ftux = fft2(u(:,:,1));
#   Ftuy = fft2(u(:,:,2));

#   fux1=reshape(Ftux, i_max*j_max,1);
#   fuy1=reshape(Ftuy, i_max*j_max,1);

#   fuu(1:2:size(fux1)*2,1) = fux1;
#   fuu(2:2:size(fuy1)*2,1) = fuy1;
  
# end