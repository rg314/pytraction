import glob
import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

x, y, u, v = pd.read_csv('data/sample.csv').T.values

plt.quiver(x,y, u,v)
plt.show()

# from skimage import io
# from pytraction.utils import allign_slice

# from openpiv import tools, pyprocess, validation, filters, scaling 

# # get sample data from Aki 

# # point and search the correct path
# sep = os.sep
# files = glob.glob(f'building{sep}Aki data{sep}PositionNTC310kPa19{sep}*')

# # get img, cell outline, and piv files 
# image = [x for x in files if '.tif' in x][0]
# outline = [x for x in files if '.csv' in x][0]
# piv_aki = [x for x in files if '.txt' in x][0]

# # read in the files and format to known format / shape
# outline = pd.read_csv(outline).values # reads in x and y as [x,y]
# outline = outline.reshape((-1,1,2)).astype(np.int32)
# img = cv2.imread(image, 0)
# piv_aki = pd.read_csv(piv_aki, delimiter='\s', header=None, engine='python')


# # load data from ryan
# rg_file = glob.glob('data/*')

# img1 = [x for x in rg_file if 'ref' not in x][0]
# ref1 = [x for x in rg_file if 'ref' in x][0]

# img1 = io.imread(img1)
# ref1 = io.imread(ref1)

# frame = 0
# channel = 0

# frame_a= np.array(img1[frame, channel, :, :], dtype='uint8')
# frame_b = np.array(ref1[channel,:,:], dtype='uint8')


# print(frame_a.shape)
# frame_a = allign_slice(frame_a, frame_b)
# print(frame_a.shape)

# img = np.stack([frame_b, frame_a])

# # io.imsave('test.tif',img)
