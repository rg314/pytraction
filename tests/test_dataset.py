from pytraction.dataset import Dataset

import os
import pandas as pd

def test_Dataset_load():

    dataset_path = os.path.join('tests', 'data', 'example_dataset.h5')

    dataset = Dataset(dataset_path)


    assert len(dataset) == 1
    assert list(dataset.columns) == [
                        'L', 
                        'beta', 
                        'cell_roi', 
                        'force_field', 
                        'frame', 
                        'mask_roi', 
                        'pos', 
                        'stack_bead_roi', 
                        'traction_map', 
                        'vec',
    ]

    assert isinstance(dataset[0], pd.core.frame.DataFrame)

    for col in dataset.columns:
        assert isinstance(dataset[col], pd.core.frame.DataFrame)


def test_Dataset_metadata():

    dataset_path = os.path.join('tests', 'data', 'example_dataset.h5')

    dataset = Dataset(dataset_path)

    assert hasattr(dataset, 'metadata')

    metadata = dataset.metadata()

    assert isinstance(metadata, dict)

    keys = [
        'E', 
        'coarse_factor', 
        'dt', 
        'meshsize', 
        'min_window_size', 
        'nb_iter_max', 
        'overlap_ratio', 
        'pix_per_mu', 
        's', 
        'sig2noise_method', 
        'tolerance', 
        'trust_1st_iter', 
        'validation_iter', 
        'validation_method',
        ]


    assert len(metadata) == len(keys)

    for key in keys:
        assert key in metadata.keys()



def test_Dataset_save():

    dataset_path = os.path.join('tests', 'data', 'example_dataset.h5')

    dataset = Dataset(dataset_path)

    assert dataset.save('test.h5') == True


def test_Dataset_str():


    dataset_path = os.path.join('tests', 'data', 'example_dataset.h5')

    dataset = Dataset(dataset_path)

    assert isinstance(dataset.__str__(), str)
    assert isinstance(dataset.__repr__(), str)

test_Dataset_str()
test_Dataset_metadata()
test_Dataset_load()