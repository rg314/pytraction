from pytraction.core import TractionForce
from pytraction.utils import plot

from skimage import io
import numpy as np
import matplotlib.pyplot as plt

# # # ######### Example 1
pix_per_mu = 1.3
E = 100 # Young's modulus in Pa

img_path = 'data/sim/tfm.tif'
ref_path = 'data/sim/tfm-ref.tif'

traction_obj = TractionForce(pix_per_mu, E=E)

img, ref, _ = traction_obj.load_data(img_path, ref_path)

x = np.array([200, 200, 1024-200, 1024-200])
y = np.array([1024-200, 200, 1024-200, 200])




log = traction_obj.process_stack(img, ref, roi=(x,y))

for frame in range(len(log)):
    plot(log, frame=frame)
    plt.show()

# from skimage import io
# io.imsave('data/sim/stack.tif', img)