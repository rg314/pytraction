from pytraction import TractionForceConfig
from pytraction import plot, process_stack
import matplotlib.pyplot as plt

pix_per_mu = 1.3 # The number of pixels per micron 
E = 100 # Youngs modulus in Pa

img_path = 'data/example1/e01_pos1_axon1.tif'
ref_path = 'data/example1/e01_pos1_axon1_ref.tif'
config = 'config/config.yaml'

traction_config = TractionForceConfig(E=E, scaling_factor=pix_per_mu, config=config)
img, ref, _ = traction_config.load_data(img_path, ref_path)
log = process_stack(img[:1,:,:,:], ref, config=traction_config)

plot(log, frame=0)
plt.show()