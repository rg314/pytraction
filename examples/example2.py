import matplotlib.pyplot as plt

from pytraction import TractionForceConfig, plot, process_stack

pix_per_mu = 9.8138
E = 1000  # Young's modulus in Pa

img_path = "data/example2/1kPa-2-Position006.tif"
ref_path = "data/example2/1kPa-2-Position006_ref.tif"
roi_path = "data/example2/1kPa-2-Position006.roi"
config = "config/config.yaml"

traction_config = TractionForceConfig(E=E, scaling_factor=pix_per_mu, config=config)
img, ref, roi = traction_config.load_data(img_path, ref_path, roi_path)
log = process_stack(img[:1, :, :, :], ref, config=traction_config, roi=roi, crop=True)

plot(log, frame=0)
plt.show()
