import glob
import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sep = os.sep
files = glob.glob(f'building{sep}Aki data{sep}PositionNTC310kPa19{sep}*')

image = [x for x in files if '.tif' in x][0]
outline = [x for x in files if '.csv' in x][0]
piv = [x for x in files if '.txt' in x][0]


outline = pd.read_csv(outline).values # reads in x and y as [x,y]
outline = outline.reshape((-1,1,2)).astype(np.int32)
img = cv2.imread(image, 0)

piv = pd.read_csv(piv, delimiter='\s', header=None)

print(img.shape)


# img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# img = cv2.drawContours(img, outline, -1, (0,255,0), 3)

