import numpy as np
from pytraction.preprocess import (
    _get_reference_frame,
    _get_img_frame,
    _get_cell_img,
    _get_raw_frames,
    _get_min_window_size,
    _load_frame_roi,
    _cnn_segment_cell,
    _detect_cell_instances_from_segmentation,
    _located_most_central_cell,
    _predict_roi,
    _get_polygon_and_roi,
    _create_mask,
    _crop_roi,
    _create_crop_mask_targets,
    )

def test_get_reference_frame():
    # 1 channel
    ref_stack1 = np.random.random((1,500,500))
    bead_channel1 = 0
    
    # 2 channel
    ref_stack2 = np.random.random((2,500,500))
    bead_channel2 = 1

    # 3 channel
    ref_stack3 = np.random.random((3,500,500))
    bead_channel3 = 2

    # 4 channel
    ref_stack4 = np.random.random((4,500,500))
    bead_channel4 = 3

    ref1 = _get_reference_frame(ref_stack1, bead_channel1)
    ref2 = _get_reference_frame(ref_stack2, bead_channel2)
    ref3 = _get_reference_frame(ref_stack3, bead_channel3)
    ref4 = _get_reference_frame(ref_stack4, bead_channel4)

    assert ref1.shape == (500,500)
    assert ref2.shape == (500,500)
    assert ref3.shape == (500,500)
    assert ref4.shape == (500,500)

    assert ref1.dtype == 'uint8'
    assert ref2.dtype == 'uint8'
    assert ref3.dtype == 'uint8'
    assert ref4.dtype == 'uint8'

def test_get_img_frame():
    # 1 channel
    img_stack1 = np.random.random((10,1,500,500))
    bead_channel1 = 0
    frame1 = 1
    
    # 2 channel
    img_stack2 = np.random.random((10, 2,500,500))
    bead_channel2 = 1
    frame2 = 1

    # 3 channel
    img_stack3 = np.random.random((10, 3,500,500))
    bead_channel3 = 2
    frame3 = 3

    # 4 channel
    img_stack4 = np.random.random((10, 4,500,500))
    bead_channel4 = 3
    frame4 = 3



    img1 = _get_img_frame(img_stack1, frame1, bead_channel1)
    img2 = _get_img_frame(img_stack2, frame2, bead_channel2)
    img3 = _get_img_frame(img_stack3, frame3, bead_channel2)
    img4 = _get_img_frame(img_stack4, frame4, bead_channel2)

    assert img1 .shape == (500,500)
    assert img2.shape == (500,500)
    assert img3.shape == (500,500)
    assert img4.shape == (500,500)

    assert img1.dtype == 'uint8'
    assert img2.dtype == 'uint8'
    assert img3.dtype == 'uint8'
    assert img4.dtype == 'uint8'



def test_get_cell_img():
    # 1 channel
    img_stack1 = np.random.random((10,1,500,500))
    cell_channel1 = 0
    frame1 = 1
    
    # 2 channel
    img_stack2 = np.random.random((10, 2,500,500))
    cell_channel2 = 1
    frame2 = 1

    # 3 channel
    img_stack3 = np.random.random((10, 3,500,500))
    cell_channel3 = 2
    frame3 = 3

    # 4 channel
    img_stack4 = np.random.random((10, 4,500,500))
    cell_channel4 = 3
    frame4 = 3

    img1 = _get_cell_img(img_stack1, frame1, cell_channel1)
    img2 = _get_cell_img(img_stack2, frame2, cell_channel2)
    img3 = _get_cell_img(img_stack3, frame3, cell_channel3)
    img4 = _get_cell_img(img_stack4, frame4, cell_channel4)

    assert img1 .shape == (500,500)
    assert img2.shape == (500,500)
    assert img3.shape == (500,500)
    assert img4.shape == (500,500)

    assert img1.dtype == 'uint8'
    assert img2.dtype == 'uint8'
    assert img3.dtype == 'uint8'
    assert img4.dtype == 'uint8'


test_get_cell_img()
test_get_img_frame()
test_get_reference_frame()


def test_get_raw_frames():
    pass
def test_get_min_window_size():
    pass
def test_load_frame_roi():
    pass
def test_cnn_segment_cell():
    pass
def test_detect_cell_instances_from_segmentation():
    pass
def test_located_most_central_cell():
    pass
def test_predict_roi():
    pass
def test_get_polygon_and_roi():
    pass
def test_create_mask():
    pass
def test_crop_roi():
    pass
def test_create_crop_mask_targets():
    pass