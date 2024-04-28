import io
import os
import h5py
import numpy as np
import pandas as pd


class TractionForceDataset(object):
    """
    Wrapper class for h5py format to write and read TFM data structures.
    """
    def __init__(self, log):
        """
        @param  log: System path to hdf5 file.
        """
        if isinstance(log, str):
            log = io.BytesIO(self.load(log))
        self.log = log
        self.columns = self._columns()

    def __str__(self):
        df = self.__getitem__(0)
        return df.__str__()

    def __repr__(self):
        df = self.__getitem__(0)
        return df.__str__()

    def __len__(self):
        with h5py.File(self.log) as f:
            length = list(f["frame"].keys())
        return len(length)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            if idx > self.__len__():
                msg = "Index out of range."
                raise IndexError(msg)
            with h5py.File(self.log) as f:
                row = {
                    x: [np.array(f[f"{x}/{idx}"])]
                    for x in f.keys()
                    if "metadata" not in x
                }
            return pd.DataFrame(row)
        elif isinstance(idx, str):
            with h5py.File(self.log) as f:
                items = {f"{idx}": []}
                for i in range(self.__len__()):
                    items[idx].append(np.array(f[f"{idx}/{i}"]))
            return pd.DataFrame(items)

    def frame_indices(self):
        with h5py.File(self.log) as f:
            indices = list(f["frame"].keys())
        return indices

    def _columns(self):
        return self.__getitem__(0).columns

    def metadata(self):
        with h5py.File(self.log) as f:
            metadata = {
                x: f["metadata"].attrs[x].tobytes() for x in f["metadata"].attrs.keys()
            }
        return metadata

    def save(self, filename):
        with open(filename, "wb") as f:
            f.write(self.log.getvalue())
        if os.path.exists(filename):
            return True
        else:
            return False

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            log = f.read()
        return log


def write_tfm_results(
        h5py_file: type(h5py.File),
        frame: int,
        pos: np.ndarray,
        pos_interp: np.ndarray,
        vec_u: np.ndarray,
        vec_u_interp: np.ndarray,
        vec_f: np.ndarray,
        beta: float,
        lamd_opt: float,
        drift_corr_stack: np.stack,
        cell_img: np.ndarray,
        mask: np.ndarray,
        config: dict
) -> type(h5py.File):
    """
    Write frame-specific TFM results to a h5py file.

    @param  h5py_file: Writeable h5py file.
    @param  frame: Time-frame number in image stack.
    @param  pos: Array containing PIV positional coordinates of deformation field.
    @param  pos_interp: Array containing interpolated positional coordinates of deformation field.
    @param  vec_u: Array containing PIV vector components of deformation field.
    @param  vec_u_interp: Array containing interpolated vector components of deformation field.
    @param  vec_f: Array containing vector components of traction field.
    @param  beta: Quantification of noise level in raw data.
    @param  lamd_opt: Regularization parameter used in solving the inverse problem.
    @param  drift_corr_stack: Drift corrected stack of deformation and reference bead image.
    @param  cell_img: Bright-field image of cell.
    @param  mask: Mask from ROI or automatic cell segmentation.
    @param  config: Config dictionary for TFM analysis.
    """

    h5py_file[f"frame/{frame}"] = frame
    h5py_file[f"position/{frame}"] = pos
    h5py_file[f"position_interpolated/{frame}"] = pos_interp
    h5py_file[f"deformation/{frame}"] = vec_u
    h5py_file[f"deformation_interpolated/{frame}"] = vec_u_interp
    h5py_file[f"traction/{frame}"] = vec_f
    h5py_file[f"noise_beta/{frame}"] = beta
    h5py_file[f"optimal_lambda/{frame}"] = lamd_opt
    h5py_file[f"drift_corrected_stack/{frame}"] = drift_corr_stack
    h5py_file[f"cell_image/{frame}"] = cell_img
    h5py_file[f"mask_roi/{frame}"] = 0 if mask is None else mask

    # Write metadata from the config file to the h5py file
    h5py_file = write_tfm_metadata(h5py_file, config)

    return h5py_file


def write_tfm_metadata(h5py_file: type(h5py.File), config: dict) -> type(h5py.File):
    """
    Write metadata of TFM config file to the h5py result file.

    @param  h5py_file: Writeable h5py file.
    @param  config: Config file for TFM analysis.
    """
    if "metadata" not in h5py_file:
        h5py_file["metadata"] = 0

    # Iterate through the PIV and TFM configuration parameters and store them in the h5py file
    for k, v in config["piv"].items():
        h5py_file["metadata"].attrs[k] = np.void(str(v).encode())

    for k, v in config["tfm"].items():
        h5py_file["metadata"].attrs[k] = np.void(str(v).encode())

    return h5py_file
