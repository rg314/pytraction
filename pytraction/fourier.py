from pytraction.interp_vec2grid import interp_vec2grid


def fourier_xu(pos, vec, meshsize, E, s, grid_mat):
    
    new_pos = pos + vec # shifted positions of data by displacements
    
    # interpolate data onto rectangular grid. If grid_mat is empty, a new grid
    # will be constructed. Otherwise the grid in grid_mat will be used.
    grid_mat,u, i_max,j_max = interp_vec2grid(new_pos, vec, meshsize, grid_mat)

#   %construct wave vectors
#   kx_vec = 2*pi/i_max/meshsize.*[0:(i_max/2-1) (-i_max/2:-1)];
#   ky_vec = 2*pi/j_max/meshsize.*[0:(j_max/2-1) (-j_max/2:-1)];
#   kx = repmat(kx_vec',1,j_max);
#   ky = repmat(ky_vec,i_max,1);
    
#   %We ignore DC component below and can therefore set k(1,1) =1
#   kx(1,1) = 1;
#   ky(1,1) = 1;
#   k = sqrt(kx.^2+ky.^2);
  
#   %calculate Green's function
#   conf = 2.*(1+s)./(E.*k.^3);
#   Gf_xx = conf .* ((1-s).*k.^2+s.*ky.^2);
#   Gf_xy = conf .* (-s.*kx.*ky);
#   Gf_yy = conf .* ((1-s).*k.^2+s.*kx.^2);

#   %set DC component to one
#   Gf_xx(1,1) = 0;
#   Gf_yy(1,1) = 0;
#   Gf_xy(1,1) = 0;
  
#   %FT of the real matrix is symmetric
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