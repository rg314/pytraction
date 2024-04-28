import os
import numpy as np
import tifffile

from pytraction.core import (TractionForceConfig, process_stack)
from pytraction.tractionforcedataset import TractionForceDataset


def test_config():
    elastic_modulus = 100
    scaling_factor = 1.3
    poisson_ratio = 0.48
    window_size = 16
    meshsize = 8
    config_path = os.path.join("example_data", "example_config.yaml")

    # Create config instance
    config = TractionForceConfig(
        config_path=config_path,
        elastic_modulus=elastic_modulus,
        scaling_factor=scaling_factor,
        poisson_ratio=poisson_ratio,
        window_size=window_size,
        meshsize=meshsize
    )
    assert config.config["tfm"]["elastic_modulus"] == elastic_modulus
    assert config.config["tfm"]["scaling_factor"] == scaling_factor
    assert config.config["tfm"]["poisson_ratio"] == poisson_ratio
    assert config.config["piv"]["window_size"] == window_size
    assert config.config["tfm"]["meshsize"] == meshsize

    # No models loaded
    assert config.cnn is None
    assert config.pre_fn is None
    assert config.knn is None

    # Models loaded (segment as kwargs)
    config = TractionForceConfig(
        config_path=config_path,
        elastic_modulus=elastic_modulus,
        scaling_factor=scaling_factor,
        poisson_ratio=poisson_ratio,
        window_size=0,
        meshsize=meshsize,
        segment=True
    )
    assert config.cnn is not None
    assert config.pre_fn is not None
    assert config.knn is not None
    assert hasattr(config.knn, "predict")


def test_load_data():
    elastic_modulus = 100
    scaling_factor = 1.3
    poisson_ratio = 0.48
    window_size = 16
    meshsize = 8
    config_path = os.path.join("example_data", "example_config.yaml")

    # Create config instance
    config = TractionForceConfig(
        config_path=config_path,
        elastic_modulus=elastic_modulus,
        scaling_factor=scaling_factor,
        poisson_ratio=poisson_ratio,
        window_size=window_size,
        meshsize=meshsize
    )
    # Test 2D image data
    # Write (t,c,w,h) image and (c,w,h) reference stacks
    img_stack = np.full((3, 2, 100, 100), 255, dtype=np.uint8)
    img_path = os.path.join("example_data", "example6_simplestacks", "white_t3c2w100h100_stack.tiff")
    with tifffile.TiffWriter(img_path) as tiff: tiff.write(img_stack)

    ref_stack = np.full((2, 100, 100), 255, dtype=np.uint8)
    ref_path = os.path.join("example_data", "example6_simplestacks", "white_c2w100h100_stack.tiff")
    with tifffile.TiffWriter(ref_path) as tiff: tiff.write(ref_stack)

    # Load image data
    img, ref, roi = config.load_data(img_path, ref_path, roi_path=None, z_proj=False)
    assert np.array_equal(img_stack, img)
    assert np.array_equal(ref_stack, ref)

    img, ref, roi = config.load_data(img_path, ref_path, roi_path=None, z_proj=True)
    assert np.array_equal(img_stack, img)
    assert np.array_equal(ref_stack, ref)

    # Test z-stack image data
    # Write (t,z,c,w,h) image and (z,c,w,h) reference stacks
    img_stack_w = np.full((4, 2, 100, 100), 255, dtype=np.uint8)
    img_stack_b = np.full((4, 2, 100, 100), 0, dtype=np.uint8)
    img_stack = np.stack((img_stack_w, img_stack_b, img_stack_w), axis=1)
    img_path = os.path.join("example_data", "example6_simplestacks", "multi_t4z3c2w100h100_stack.tiff")
    with tifffile.TiffWriter(img_path) as tiff: tiff.write(img_stack)

    ref_stack_w = np.full((2, 100, 100), 255, dtype=np.uint8)
    ref_stack_b = np.full((2, 100, 100), 0, dtype=np.uint8)
    ref_stack = np.stack((ref_stack_w, ref_stack_b, ref_stack_w), axis=0)
    ref_path = os.path.join("example_data", "example6_simplestacks", "multi_z3c2w100h100_stack.tiff")
    with tifffile.TiffWriter(ref_path) as tiff: tiff.write(ref_stack)

    # Load image data and test central z-slice extraction
    img, ref, roi = config.load_data(img_path, ref_path, roi_path=None, z_proj=False)
    assert np.array_equal(img_stack_b, img)
    assert np.array_equal(ref_stack_b, ref)

    # Load image data and test z-projection
    img, ref, roi = config.load_data(img_path, ref_path, roi_path=None, z_proj=True)
    assert np.array_equal(img_stack_w, img)
    assert np.array_equal(ref_stack_w, ref)

    # Load ROI
    img_path = os.path.join("example_data", "example2_fibroblast", "2DTFM_10kPa_hPAAGel_3T3Fibroblasts_TimeSeries.tif")
    ref_path = os.path.join("example_data", "example2_fibroblast", "2DTFM_10kPa_hPAAGel_3T3Fibroblasts_Reference.tif")
    roi_path = os.path.join("example_data", "example2_fibroblast", "AllRoiFrames.zip")

    img, ref, roi = config.load_data(img_path, ref_path, roi_path, z_proj=False)

    assert roi is not None


def test_process_stack():
    elastic_modulus = 100
    scaling_factor = 1.3
    poisson_ratio = 0.48
    window_size = 32
    meshsize = 16
    config_path = os.path.join("example_data", "example_config.yaml")

    # Create config instance
    config = TractionForceConfig(
        config_path=config_path,
        elastic_modulus=elastic_modulus,
        scaling_factor=scaling_factor,
        poisson_ratio=poisson_ratio,
        window_size=window_size,
        meshsize=meshsize
    )

    img_path = os.path.join("example_data", "example3_zstack", "2DTFM_300Pa_PAAGel_RGCs_TimeSeries.tif")
    ref_path = os.path.join("example_data", "example3_zstack", "2DTFM_300Pa_PAAGel_RGCs_Reference.tif")

    img, ref, roi = config.load_data(img_path=img_path, ref_path=ref_path, roi_path=None, z_proj=True)

    log = process_stack(img_stack=img,
                        ref_stack=ref,
                        roi=roi,
                        config=config,
                        cell_channel=0,
                        bead_channel=1,
                        crop=False,
                        noise=10)

    assert type(log) is TractionForceDataset
