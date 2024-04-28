import matplotlib.pyplot as plt
import numpy as np
import imageio
import io

from typing import Tuple, Type, Union
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.integrate import simps
from pytraction.tractionforcedataset import TractionForceDataset
from pytraction.utils import interp_mask2grid


def tfm_plot(
        tfm_dataset: type(TractionForceDataset),
        vec_field: str,
        frames: Union[list, int],
        heat_map: Union[bool, np.ndarray] = True,
        mask: Union[bool, np.ndarray] = False,
        vmax: float = None,
        figsize: tuple = (16, 16),
        cbar_label: str = "Traction stress [Pa]"
):
    """
    Plots 2D vector field stored in TractionForceDataset on top of traction field heatmap or cell image.

    @param  tfm_dataset: TractionForceDataset file containing TFM results.
    @param  vct_field: Vector field to be drawn. Options include "piv_disp" (PIV displacement field), "int_disp"
    (interpolated displacement field) and "trct" (traction field).
    @param  frames: List of numbers indicating which frames are used.
    @param  heat_map: Set heat map of traction magnitude as background (True) or use cell image (False). If scalar field
    is provided, it will be used as background.
    @param  mask: Whether to use mask in plot or not.
    @param  vmax: Maximum value of heatmap colorbar, not used if heat_map set to False.
    @param  figsize: Size (width, height) of plot in inch.
    @param  cbar_label: Specify title displayed on y-axis
    """

    # Get full identifier names for corresponding vector field
    if vec_field == "piv_disp":
        vec_field = "deformation"
        pos_field = "position"
    elif vec_field == "int_disp":
        vec_field = "deformation_interpolated"
        pos_field = "position_interpolated"
    elif vec_field == "trct":
        vec_field = "traction"
        pos_field = "position_interpolated"
    else:
        msg = (f"{vec_field} is not an option for vct_field, please use one of the following: piv_disp, int_disp or"
               f"trct.")
        raise RuntimeError(msg)

    if isinstance(frames, int):
        frames = np.asarray([frames], dtype=int)

    # Create empty arrays
    pos, vec, vec_dim, corr_stack, cell_img = [], [], [], [], []

    # Find the most common image dimension in time-series
    for frame in frames:
        frame = int(frame)
        dim_tmp = (tfm_dataset[frame][vec_field][0]).shape
        # Append dimension of frames vector field to list
        vec_dim.append(dim_tmp[:2])

    # Get most occurring dimension tuple
    comm_dim = max(set(vec_dim), key=vec_dim.count)

    # Extract time-series results frame-by-frame
    for frame in frames:
        frame = int(frame)
        tfm_frame = tfm_dataset[frame]

        # Filter images with dimensions different from most common dimension
        if (tfm_frame[vec_field][0]).shape[:2] != comm_dim:
            continue

        pos.append(tfm_frame[pos_field][0])
        vec.append(tfm_frame[vec_field][0])
        corr_stack.append(tfm_frame["drift_corrected_stack"][0])
        cell_img.append(tfm_frame["cell_image"][0])

    # Calculate mean fields from time-series frames
    pos = np.mean(pos, axis=0)
    vec = np.mean(vec, axis=0)

    pos_x, pos_y = pos[:, :, 0], pos[:, :, 1]
    vec_x, vec_y = vec[:, :, 0], vec[:, :, 1]

    # Create plots
    fig, ax = plt.subplots(1, 2, figsize=figsize)

    if heat_map is False:
        # Plot background cell image
        ax[0].imshow(
            cell_img[0],
            cmap="gray",
            alpha=0.6,
            extent=[pos_x.min(), pos_x.max(), pos_y.min(), pos_y.max()],
        )
        ax[0].set_axis_off()

        # Plot vector field
        main_clr = ax[0].quiver(pos_x, pos_y, vec_x, vec_y, np.sqrt(vec_x ** 2 + vec_y ** 2), cmap='jet')
    else:
        if heat_map is True:
            # Calculate background traction map
            scalar_map = np.sqrt(vec_x ** 2 + vec_y ** 2)
            scalar_map = np.flipud(scalar_map)
        elif type(heat_map) is np.ndarray:
            scalar_map = heat_map
        else:
            msg = f"{type(heat_map)} is not implemented for heat_map variable, please use bool or np.ndarray types."
            raise RuntimeError(msg)

        # Set vmax value
        vmax = np.max(scalar_map) if not vmax else vmax

        # Plot background traction map
        main_clr = ax[0].imshow(
            scalar_map,
            interpolation="bicubic",
            cmap="jet",
            extent=[pos_x.min(), pos_x.max(), pos_y.min(), pos_y.max()],
            vmin=0,
            vmax=vmax,
        )
        ax[0].set_axis_off()

        # Plot vector field
        ax[0].quiver(pos_x, pos_y, vec_x, vec_y)

    # Plot mask
    tfm_frame = tfm_dataset[0]
    if mask is False:
        mask = np.full((tfm_frame["cell_image"][0]).shape, 255)
        mask = np.ma.masked_where(mask == 255, mask)
    elif mask is True:
        mask = tfm_frame["mask_roi"][0]
        mask = np.ma.masked_where(mask == 255, mask)
    elif type(mask) is not np.ndarray:
        msg = f"{type(mask)} is not implemented for the mask variable, please use bool or np.ndarray types."
        raise RuntimeError(msg)
    ax[0].imshow(mask, cmap="jet", extent=[pos_x.min(), pos_x.max(), pos_y.min(), pos_y.max()])

    # Plot colorbar
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(main_clr, cax=cax)
    cbar.set_label(cbar_label, rotation=270, labelpad=35, size=25)
    cbar.ax.tick_params(labelsize=20)
    ax[1].remove()

    return fig, ax


def tfm_savegif(figs: list, sys_path: str, fps: float = 5):
    """
    Creates .gif movie from matplotlib figures with provided frames per second.
    """
    # Set colorbar maximum of all figures equally to the largest
    # figs, vmax = set_cbar_max(figs)  # ToDo: Fix

    images = []
    for fig in figs:
        # Save images to buffer as .png
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=fig.dpi, bbox_inches="tight")

        # Rewind the pointer to beginning of file and read
        buffer.seek(0)
        image = imageio.imread(buffer)
        images.append(image)

    # Save the images to a GIF file
    imageio.mimsave(sys_path, images, duration=1/fps)


def strain_energy(tfm_dataset: type(TractionForceDataset),
                  frame: int,
                  mask: Union[bool, np.ndarray] = False):
    """
    Calculates strain energy of the displacement and traction field in the spatial domain.
    """
    tfm_frame = tfm_dataset[int(frame)]

    if mask is False:
        mask = np.full((tfm_frame["position_interpolated"][0]).shape, 255)
        mask = np.where(mask == 255, True, False)
    elif mask is True:
        mask = np.flipud(interp_mask2grid(mask=tfm_frame["mask_roi"][0], pos=tfm_frame["position_interpolated"][0]))
        mask = np.stack((mask, mask), 2)
    elif type(mask) is not np.ndarray:
        msg = f"{type(mask)} is not implemented for the mask variable, please use bool or np.ndarray types."
        raise RuntimeError(msg)

    pos = tfm_frame["position_interpolated"][0]
    vec_u = tfm_frame["deformation_interpolated"][0] * mask
    vec_f = tfm_frame["traction"][0] * mask

    # Calculate inner product of traction and displacement field
    energy_dens = np.flipud(vec_f[:, :, 0] * vec_u[:, :, 0] + vec_f[:, :, 1] * vec_u[:, :, 1])

    # Flatten vectors to define integration intervals spaced accordingly to grid
    x = pos[0, :, 0].reshape(1, -1).flatten()
    y = pos[:, 0, 1].reshape(1, -1).flatten()

    # Integrate energy density over whole domain
    # For displacements in micro-meter and tractions in Pa, this yields an energy in pico Joule (pN/m) units
    energy = 0.5 * simps(simps(energy_dens, y), x)

    return energy_dens, energy
