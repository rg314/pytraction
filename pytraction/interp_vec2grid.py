import numpy as np 

def interp_vec2grid(pos, vec, cluster_size, grid_mat):
    if not grid_mat:
        max_eck = [np.max(pos[0]), np.max(pos[1])]
        min_eck = [np.min(pos[0]), np.min(pos[1])]

        i_max = np.floor((max_eck[0]-min_eck[0])/cluster_size)
        j_max = np.floor((max_eck[1]-min_eck[1])/cluster_size)
        
        i_max = i_max - np.mod(i_max,2)
        j_max = j_max - np.mod(j_max,2)

        X = min_eck[0] + np.arange(0.5, i_max, 0.5)*cluster_size
        Y = min_eck[1] + np.arange(0.5, j_max, 0.5)*cluster_size

        x, y = np.meshgrid(X, Y)

        pass
#         [X,Y] = meshgrid(min_eck(1)+(1/2:1:(i_max))*cluster_size, min_eck(2)+(1/2:1:(j_max))*cluster_size);
#         grid_mat(:,:,1) = X';
#         grid_mat(:,:,2) = Y';
#         clear X Y;
      
#     else
#         i_max = size(grid_mat,1);
#         j_max = size(grid_mat,2);
#         cluster_size = grid_mat(1,1,1) - grid_mat(2,2,1);
#     end
    
#     if any(isnan(vec))
#         disp('Warning: original data contains NAN values. Removing these values!');
#         pos(isnan(vec(:,1)) | isnan(vec(:,2)),:) = [];
#         vec(isnan(vec(:,1)) | isnan(vec(:,2)),:) = [];
#     end
    
#     u(1:i_max,1:j_max,1) = griddata(pos(:,1),pos(:,2),vec(:,1),grid_mat(:,:,1),grid_mat(:,:,2),'cubic');
#     u(1:i_max,1:j_max,2) = griddata(pos(:,1),pos(:,2),vec(:,2),grid_mat(:,:,1),grid_mat(:,:,2),'cubic');
#     u(isnan(u)) = 0;    
# end

    return grid_mat,u, i_max, j_max