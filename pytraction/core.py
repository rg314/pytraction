import io
import os
import pickle
import h5py
import numpy as np
import segmentation_models_pytorch as smp
import tifffile
import torch
import yaml

from typing import Tuple, Type, Union, Any
from shapely import geometry
from pytraction.tractionforcedataset import TractionForceDataset
from pytraction.net.dataloader import get_preprocessing
from pytraction.preprocess import get_min_window_size, get_raw_frames
from pytraction.process import calculate_traction_map, iterative_piv, interp_vec2grid
from pytraction.roi import roi_loaders, load_frame_roi, create_crop_mask_targets, get_polygon_and_roi
from pytraction.noise import get_noise
from pytraction.tractionforcedataset import write_tfm_results


class TractionForceConfig(object):
    """
    Configuration class for 2D traction force microscopy analysis.
    Inherits from 'object' class (default).
    """

    def __init__(
            self,
            config_path: str,
            elastic_modulus: float,
            scaling_factor: float,
            poisson_ratio: Union[float, None] = None,
            window_size: Union[int, None] = None,
            meshsize: Union[float, None] = None,
            **kwargs
    ):
        """
        @param  config_path: System path to config.yaml file.
        @param  elastic_modulus: Young's modulus of culture substrate in Pa.
        @param  scaling_factor: Number of pixels per micrometer in x-y plane.
        @param  poisson_ratio: Poisson's ratio of culture substrate. If not set, uses value from config file.
        @param  window_size: Must be multiple of base 2 i.e. 8, 16, 32, 64. Determines the size of the subregions
        used for tracking particle motion which should be adjusted to bead density of the input images. If not set,
        value will be predicted from bead density using KNN class (Default).
        @param  meshsize: Specifies number of grid intervals to interpolate displacement field on. To keep PIV
        resolution, set to overlap_ratio * window_size. If not set, uses value from config file.
        @param **kwargs
        """
        # Load and configure parameters from the YAML config file
        self.config = self._config_yaml(config_path=config_path,
                                        elastic_modulus=elastic_modulus,
                                        scaling_factor=scaling_factor,
                                        poisson_ratio=poisson_ratio,
                                        window_size=window_size,
                                        meshsize=meshsize)

        # Set additional settings parameters by iterating over key-value pairs
        for k, v in kwargs.items():
            self.config["settings"][k] = v

        if (self.config["piv"]["window_size"] is None) or (self.config["piv"]["window_size"] == 0):
            # Load K-nearest neighbors model (KNN) to predict minimum window size based on bead density
            self.knn = self._get_knn_model()
        else:
            self.knn = None

        if self.config["settings"]["segment"] is True:
            # Load convolutional neural network (CNN) model with preprocessing function to predict cell contour in image
            self.cnn, self.pre_fn = self._get_cnn_model(device=self.config["settings"]["device"])
        else:
            self.cnn, self.pre_fn = (None, None)

    def __repr__(self):
        """
        Custom representation of the object, when called by the built-in repr() function.
        """
        pass

    @staticmethod
    def _config_yaml(
            config_path: str,
            elastic_modulus: float,
            scaling_factor: float,
            poisson_ratio: Union[float, None],
            window_size: Union[int, None],
            meshsize: Union[float, None]
    ) -> dict:
        """
        Import config.yaml file from specified system path and parse to a python dictionary.
        """
        with open(config_path, "r") as config_file:
            # Parse yaml file to python dictionary
            config = yaml.load(stream=config_file, Loader=yaml.FullLoader)

        # Overwrite parts of imported config dictionary with user input data
        config["tfm"]["elastic_modulus"] = elastic_modulus
        config["tfm"]["scaling_factor"] = scaling_factor
        config["tfm"]["poisson_ratio"] = (
            poisson_ratio if poisson_ratio is not None else config["tfm"]["poisson_ratio"]
        )
        config["piv"]["window_size"] = (
            window_size if window_size is not None else config["piv"]["window_size"]
        )
        config["tfm"]["meshsize"] = (
            meshsize if meshsize is not None else config["tfm"]["meshsize"]
        )
        return config

    @staticmethod
    def _get_cnn_model(device: str) -> Tuple[Any, Any]:
        """
        Load a Convolutional Neural Network (CNN) model.

        @param  device: Ensures that CNN model can be used on the computing device (cpu, cuda, ...).
        """
        # New path to the model file in the package directory
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'cnn_model.pth')

        # Check if the model file exists at the new path
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at {model_path}. Please make sure it is located in the 'models' folder.")

        # Load the CNN model
        # "cpu" ensures that model can be used on a CPU even if the original training was done on a different device
        cnn_model = torch.load(f=model_path, map_location="cpu")

        if device == "cuda" and torch.cuda.is_available():
            cnn_model = cnn_model.to("cuda")

        # Retrieves a preprocessing function for images from the segmentation_models_pytorch library
        preproc_fn = smp.encoders.get_preprocessing_fn("efficientnet-b1", "imagenet")
        preprocessing_fn = get_preprocessing(preproc_fn)

        return cnn_model, preprocessing_fn

    @staticmethod
    def _get_knn_model() -> Any:
        """
        Load a K-nearest neighbors (KNN) model.
        """
        # New path to the KNN model file in the package directory
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'knn_model.pickle')

        # Check if the KNN model file exists at the new path
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"KNN model file not found at {model_path}. Please make sure it's in the 'models' folder.")

        with open(model_path, "rb") as f:
            knn_model = pickle.load(f)

        return knn_model

    @staticmethod
    def load_data(
            img_path: str,
            ref_path: Union[str, None] = None,
            roi_path: Union[str, None] = None,
            z_proj: bool = True
    ):
        """
        Load image data, reference data and ROI data from given paths.

        @param  img_path: System path to image .tiff file with dimensions (t,c,w,h) or (t,z,c,w,h).
        @param  ref_path: System path to reference .tiff file with dimensions (c,w,h) or (z,c,w,h). Dynamic reference
        frame (Time-Series) will be used if set to None.
        @param  roi_path: System path to file containing ROI information (.csv, .roi or .zip). If segment is set to True
        in config file, ROI will be ignored. No ROI will be loaded if roi_path is set to None.
        @param  z_proj: Apply maximum intensity projection along z-axis for image and reference stacks.
        If set to False or dimension does not include z-axis, the center images will be used.
        """
        # Read .tiff image and store as numpy array
        img = tifffile.imread(img_path)

        if ref_path is not None:
            #  Read .tiff reference file and store as numpy array
            ref = tifffile.imread(ref_path)
        else:
            # Dynamic reference frame
            ref = None

        # Load ROI file
        roi = roi_loaders(roi_path)

        if not isinstance(img, np.ndarray) or not isinstance(ref, (np.ndarray, type(None))):
            msg = f"Image data not loaded for {img_path} or {ref_path}."
            raise TypeError(msg)

        # z-slice projection
        if len(img.shape) == 5:
            if z_proj is True:
                img = np.max(img, axis=1)
                msg = f"3D image stack was projected to a single z-plane and the new stack shape is {img.shape}."
                print(msg)
            else:
                # Select most central z-slice
                z_centre = img.shape[1] // 2
                img = np.squeeze(img[:, z_centre:z_centre + 1, :, :, :])
                msg = f"Central z-slice of 3D image stack was extracted and the new stack shape is {img.shape}."
                print(msg)
        else:
            assert len(img.shape) == 4, (f"Please ensure that the input image format is either (t,c,w,h) or "
                                         f"(t,z,c,w,h),the current shape is {img.shape}.")

        if ref is None:
            msg = f"Using dynamic reference frame for PIV calculations."
            print(msg)
        elif len(ref.shape) == 4:
            if z_proj is True:
                ref = np.max(ref, axis=0)
                msg = f"3D reference stack was projected to a single z-plane and the new stack shape is {ref.shape}."
                print(msg)
            else:
                z_centre = ref.shape[1] // 2
                ref = np.squeeze(ref[z_centre:z_centre + 1, :, :, :])
                msg = f"Central z-slice of 3D reference stack was extracted and the new stack shape is {ref.shape}."
                print(msg)
        else:
            assert len(ref.shape) == 3, (f"Please ensure that the reference image format is either (c,w,h) or "
                                         f"(z,c,w,h), the current shape is {ref.shape}.")

        return img, ref, roi


def process_stack(
        img_stack: np.ndarray,
        ref_stack: Union[np.ndarray, None],
        roi: Union[Tuple[list, list], list, None],
        config: type(TractionForceConfig),
        cell_channel: int = 0,
        bead_channel: int = 1,
        crop: bool = False,
        noise: Union[np.ndarray, Type[geometry.Polygon], int] = 10
) -> type(TractionForceDataset):
    """
    Central function to calculate PIV, traction map & save results to HDF5 file.

    @param  img_stack: Image stack as returned from load_data function.
    @param  ref_stack: Reference stack as returned from load_data function.
    @param  roi: ROI data as returned from load_data function.
    @param  config: TractionForceConfig class instance for pytraction analysis.
    @param  cell_channel: Cell channel occurrence (0 or 1).
    @param  bead_channel: Bead channel occurrence (0 or 1).
    @param  crop: Crop the image to the selected ROI with a border margin of 10%.
    @param  noise: Integer specifies height of rectangle at top border of image which marks the region of displacement
    vectors used for noise calculation. Alternatively, region can be defined more specifically by passing polygon shape
    or custom array of flattened displacement vector components.
    """
    # Check if config is instance of TractionForceConfig
    if not isinstance(config, TractionForceConfig):
        msg = f"Please verify that config is a instance of TractionForceConfig{type(config)}."
        raise TypeError(msg)

    # Determine the number of time-frames in the image stack
    if ref_stack is not None:
        n_frames = img_stack.shape[0]
    else:
        # Dynamic reference frame
        n_frames = img_stack.shape[0] - 1
        assert n_frames > 0

    # Create an in-memory binary buffer for storing results without creating a physical file
    bytes_hdf5 = io.BytesIO()

    # Open an HDF5 file for storing large and complex data structures
    with h5py.File(bytes_hdf5, "w") as h5py_file:
        # Loop through each time-frame
        for frame in list(range(n_frames)):
            # Load image, reference, and cell image in 8bit format for the current frame
            img, ref, cell_img = get_raw_frames(
                img_stack=img_stack, ref_stack=ref_stack, frame=frame, bead_channel=bead_channel,
                cell_channel=cell_channel
            )

            # Get the minimum window size for PIV
            window_size = get_min_window_size(img=img, config=config)
            config.config["piv"]["window_size"] = window_size

            # Load ROI for the current frame
            roi_i = load_frame_roi(roi=roi, frame=frame, nframes=n_frames)

            # Segment most central cell (or use ROI) to define polygon around cell
            polygon, pts = get_polygon_and_roi(cell_img=cell_img, roi=roi_i, config=config)

            # Crop targets to polygon selection
            img, ref, cell_img, mask = create_crop_mask_targets(img=img, ref=ref, cell_img=cell_img, pts=pts, crop=crop)

            # Perform PIV to calculate displacement vectors (u, v) for positions (x, y)
            x, y, u, v, dx, dy, drift_corrected_stack = iterative_piv(img=img, ref=ref, config=config)

            # Calculate noise value beta inside ROI, segmented cell or whole image
            beta = get_noise(x=x, y=y, u=u, v=v, polygon=polygon, noise=noise)

            # Create arrays for position (pos) and displacement vectors (vec)
            pos = np.stack((x, y), axis=2)
            pos_flat = np.array([x.flatten(), y.flatten()])

            vec = np.stack((u, v), axis=2)
            vec_flat = np.array([u.flatten(), v.flatten()])

            # Interpolate displacement field onto rectangular grid using meshsize
            scaled_meshsize = config.config["tfm"]["meshsize"]/config.config["tfm"]["scaling_factor"]
            pos_interp, vec_interp = interp_vec2grid(pos_flat=pos_flat,
                                                     vec_flat=vec_flat,
                                                     meshsize=scaled_meshsize)

            # Compute traction field from displacement field using Boussinesq equation and tikhonov regularization
            vec_f, lambd, evidence_one = calculate_traction_map(
                pos=pos_interp,
                vec_u=vec_interp,
                beta=beta,
                config=config
            )

            # Write results for the current frame to the open h5py file
            h5py_file = write_tfm_results(
                h5py_file=h5py_file,
                frame=frame,
                pos=pos,
                pos_interp=pos_interp,
                vec_u=vec,
                vec_u_interp=vec_interp,
                vec_f=vec_f,
                beta=beta,
                lamd_opt=lambd,
                drift_corr_stack=drift_corrected_stack,
                cell_img=cell_img,
                mask=mask,
                config=config.config,
            )

    # Convert to TFM dataset
    tfm_dataset = TractionForceDataset(bytes_hdf5)

    return tfm_dataset
