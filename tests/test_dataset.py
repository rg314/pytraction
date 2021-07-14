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


def test_Dataset_metadata():

    dataset_path = os.path.join('tests', 'data', 'example_dataset.h5')

    dataset = Dataset(dataset_path)

    assert hasattr(dataset, 'metadata')


def test_Dataset_save():

    dataset_path = os.path.join('tests', 'data', 'example_dataset.h5')

    dataset = Dataset(dataset_path)

    assert dataset.save('test.h5') == True


