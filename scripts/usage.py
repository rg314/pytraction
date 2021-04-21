from pytraction import TractionForce, plot, Dataset

from skimage import io
import numpy as np
import matplotlib.pyplot as plt

# # ######### Example 1
pix_per_mu = 1.3
E = 100 # Young's modulus in Pa

img_path = 'data/example1/e01_pos1_axon1.tif'
ref_path = 'data/example1/e01_pos1_axon1_ref.tif'

traction_obj = TractionForce(pix_per_mu, E=E)

img, ref, _ = traction_obj.load_data(img_path, ref_path)

log = traction_obj.process_stack(img[:1,:,:,:], ref)

print(log)


# # # # ########## Example 2
# pix_per_mu = 9.8138
# E = 1000 # Young's modulus in Pa

# img_path = 'data/example2/1kPa-2-Position006.tif'
# ref_path = 'data/example2/1kPa-2-Position006_ref.tif'
# roi_path = 'data/example2/1kPa-2-Position006.roi'

# traction_obj = TractionForce(pix_per_mu, E=E)

# img, ref, roi = traction_obj.load_data(img_path, ref_path, roi_path)

# log = traction_obj.process_stack(img, ref, roi=roi)

# print(log)


# # # ########## Example 3
# pix_per_mu = 1
# E = 3000 # Young's modulus in Pa

# img_path_bead = 'data/example3/Beads3.tif'
# img_path_cell = 'data/example3/Cell3.tif'
# ref_path = 'data/example3/BeadsStop.tif'

# def z_project(img_path):
#         img = io.imread(img_path)
#         img_max= np.max(img, axis=0)
#         return img_max

# bead = z_project(img_path_bead)
# cell = z_project(img_path_cell)
# ref = z_project(ref_path)

# img = np.stack([[bead, cell]])
# ref = np.stack([ref, ref])

# io.imsave('data/example3/tfm.tif', img)
# io.imsave('data/example3/tfm-ref.tif', ref)


# img_path = 'data/example3/tfm.tif'
# ref_path = 'data/example3/tfm-ref.tif'

# traction_obj = TractionForce(pix_per_mu, E=E, segment=True)

# img, ref, roi = traction_obj.load_data(img_path, ref_path)

# log = traction_obj.process_stack(img, ref, roi=roi)

# print(log)