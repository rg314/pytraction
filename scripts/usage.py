from pytraction.core import TractionForce
import matplotlib.pyplot as plt
from skimage import io
import numpy as np

######### Example 1
pix_per_mu = 1.3
E = 100 # Young's modulus in Pa

img_path = 'data/e01_pos1_axon1.tif'
ref_path = 'data/e01_pos1_axon1_ref.tif'

traction_obj = TractionForce(pix_per_mu, E=E)

img, ref, _ = traction_obj.load_data(img_path, ref_path)

log = traction_obj.process_stack(img, ref)

print(log)


########## Example 2
pix_per_mu = 9.8138
E = 1000 # Young's modulus in Pa

img_path = 'data/greenhalgh-pytraction-test-data/1kPa-2-Position007.tif'
ref_path = 'data/greenhalgh-pytraction-test-data/1kPa-2-Position007-ref.tif'
roi_path = 'data/greenhalgh-pytraction-test-data/1kPa-2-Position007.roi'

traction_obj = TractionForce(pix_per_mu, E=E)

img, ref, roi = traction_obj.load_data(img_path, ref_path, roi_path)

log = traction_obj.process_stack(img, ref, roi=roi)

print(log)


########## Example 3
pix_per_mu = 1
E = 3000 # Young's modulus in Pa

img_path_bead = 'data/aki_example/Beads3.tif'
img_path_cell = 'data/aki_example/Cell3.tif'
ref_path = 'data/aki_example/BeadsStop.tif'

def z_project(img_path):
        img = io.imread(img_path)
        img_max= np.max(img, axis=0)
        return img_max

bead = z_project(img_path_bead)
cell = z_project(img_path_cell)
ref = z_project(ref_path)

img = np.stack([[bead, cell]])
ref = np.stack([ref, ref])

io.imsave('data/aki_example/tfm.tif', img)
io.imsave('data/aki_example/tfm-ref.tif', ref)


img_path = 'data/aki_example/tfm.tif'
ref_path = 'data/aki_example/tfm-ref.tif'

traction_obj = TractionForce(pix_per_mu, E=E, segment=True, window_size=16)

img, ref, roi = traction_obj.load_data(img_path, ref_path)

log = traction_obj.process_stack(img, ref, roi=roi)


