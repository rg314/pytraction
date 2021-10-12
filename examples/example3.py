import matplotlib.pyplot as plt
import numpy as np
from skimage import io

from pytraction import TractionForceConfig, plot, process_stack

pix_per_mu = 1
E = 3000  # Young's modulus in Pa

img_path_bead = "data/example3/Beads3.tif"
img_path_cell = "data/example3/Cell3.tif"
ref_path = "data/example3/BeadsStop.tif"
config = "config/config.yaml"

# read in images
img_bead = io.imread(img_path_bead)
img_cell = io.imread(img_path_cell)
ref = io.imread(ref_path)

# reformat images
def z_project(img_path):
    img = io.imread(img_path)
    img_max = np.max(img, axis=0)
    return img_max


bead = z_project(img_path_bead)
cell = z_project(img_path_cell)
ref = z_project(ref_path)

img = np.stack([[bead, cell]])
ref = np.stack([ref, ref])

# save images in correct shapes
io.imsave("data/example3/tfm.tif", img)
io.imsave("data/example3/tfm-ref.tif", ref)


img_path = "data/example3/tfm.tif"
ref_path = "data/example3/tfm-ref.tif"

traction_config = TractionForceConfig(E=E, scaling_factor=pix_per_mu, config=config)
traction_config.config["settings"]["segment"] = True

img, ref, _ = traction_config.load_data(img_path, ref_path)
log = process_stack(img[:1, :, :, :], ref, config=traction_config)

plot(log, frame=0)
plt.show()
