import matplotlib.pyplot as plt
import numpy as np
from skimage import io

from pytraction import TractionForceConfig, plot, prcoess_stack

# # # ######### Example 1
pix_per_mu = 1.3
E = 100  # Young's modulus in Pa

img_path = "data/sim/tfm.tif"
ref_path = "data/sim/tfm-ref.tif"

traction_config = TractionForceConfig(pix_per_mu, E=E)

img, ref, _ = traction_config.load_data(img_path, ref_path)

x = np.array([200, 200, 1024 - 200, 1024 - 200])
y = np.array([1024 - 200, 200, 1024 - 200, 200])


log = process_stack(img, ref, traction_config, roi=(x, y))

for frame in range(len(log)):
    plot(log, frame=frame)
    plt.show()

# from skimage import io
# io.imsave('data/sim/stack.tif', img)
