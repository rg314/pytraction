import numpy as np

from pytraction.preprocess import (_cnn_segment_cell,
                                   _create_crop_mask_targets, _create_mask,
                                   _crop_roi,
                                   _detect_cell_instances_from_segmentation,
                                   _get_cell_img, _get_img_frame,
                                   _get_min_window_size, _get_polygon_and_roi,
                                   _get_raw_frames, _get_reference_frame,
                                   _load_frame_roi, _located_most_central_cell,
                                   _predict_roi)


def test_get_reference_frame():
    # 1 channel
    ref_stack1 = np.ones((1, 500, 500))
    bead_channel1 = 0

    # 2 channel
    ref_stack2 = np.ones((2, 500, 500))
    bead_channel2 = 1

    # 3 channel
    ref_stack3 = np.ones((3, 500, 500))
    bead_channel3 = 2

    # 4 channel
    ref_stack4 = np.ones((4, 500, 500))
    bead_channel4 = 3

    ref1 = _get_reference_frame(ref_stack1, None, bead_channel1)
    ref2 = _get_reference_frame(ref_stack2, None, bead_channel2)
    ref3 = _get_reference_frame(ref_stack3, None, bead_channel2)
    ref4 = _get_reference_frame(ref_stack4, None, bead_channel2)

    assert ref1.shape == (500, 500)
    assert ref2.shape == (500, 500)
    assert ref2.shape == (500, 500)
    assert ref2.shape == (500, 500)


def test_get_img_frame():
    pass


def test_get_cell_img():
    pass


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
