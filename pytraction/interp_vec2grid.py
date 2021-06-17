import numpy as np 
from scipy.interpolate import griddata

def interp_vec2grid(pos, vec, cluster_size, grid_mat=np.array([])):
    if not grid_mat:
        max_eck = [np.max(pos[0]), np.max(pos[1])]
        min_eck = [np.min(pos[0]), np.min(pos[1])]

        i_max = np.floor((max_eck[0]-min_eck[0])/cluster_size)
        j_max = np.floor((max_eck[1]-min_eck[1])/cluster_size)
        
        i_max = i_max - np.mod(i_max,2)
        j_max = j_max - np.mod(j_max,2)

        X = min_eck[0] + np.arange(0.5, i_max)*cluster_size
        Y = min_eck[1] + np.arange(0.5, j_max)*cluster_size

        x, y = np.meshgrid(X, Y)

        grid_mat = np.stack([x,y], axis=2)

        u = griddata(pos.T, vec.T, (x,y),method='cubic')

        return grid_mat,u, int(i_max), int(j_max)