import os


def validate_data_paths():
    # Config file
    assert os.path.join("example_data", "example_config.yaml")

    # Example: Axon
    assert os.path.join("example_data", "example1_axon", "2DTFM_300Pa_RGC_Axon_Reference.tif")
    assert os.path.join("example_data", "example1_axon", "2DTFM_300Pa_RGC_Axon_TimeSeries.tif")

    # Example: Fibroblast
    assert os.path.join("example_data", "example2_fibroblast", "2DTFM_10kPa_hPAAGel_3T3Fibroblasts_TimeSeries.tif")
    assert os.path.join("example_data", "example2_fibroblast", "2DTFM_10kPa_hPAAGel_3T3Fibroblasts_Reference.tif")
    assert os.path.join("example_data", "example2_fibroblast", "AllRoiFrames.zip")
    assert os.path.join("example_data", "example2_fibroblast", "frame0.roi")

    # Example: z-stack
    assert os.path.join("example_data", "example3_zstack", "2DTFM_300Pa_PAAGel_RGCs_TimeSeries.tif")
    assert os.path.join("example_data", "example3_zstack", "2DTFM_300Pa_PAAGel_RGCs_Reference.tif")

    # Example: Given displacement field
    assert os.path.join("example_data", "example4_displacement", "example_position.npy")
    assert os.path.join("example_data", "example4_displacement", "example_displacement.npy")

    # Example_ Given traction dataset
    assert os.path.join("example_data", "example5_tractiondataset", "example_tractiondataset.h5")
