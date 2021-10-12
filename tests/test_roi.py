from pytraction.roi import (_load_csv_roi, _load_roireader_roi, _load_zip_roi,
                            _recursive_lookup, roi_loaders)


def test__recursive_lookup():
    nested_1 = {"a": {"b": {"x": [1, 2, 3]}}}
    nested_2 = {"a": {"b": {"a": "a"}}}

    x1 = _recursive_lookup("x", nested_1)
    x2 = _recursive_lookup("x", nested_2)

    assert x1 == [1, 2, 3]
    assert x2 == None


def test__load_csv_roi():
    csv_example = "tests/data/roi.csv"

    (x, y) = _load_csv_roi(csv_example)

    assert isinstance(x, list)
    assert isinstance(y, list)
    assert len(x) == len(y)


def test__load_roireader_roi():
    imagej_roi_example = "tests/data/imagej_roi.roi"

    (x, y) = _load_roireader_roi(imagej_roi_example)

    assert isinstance(x, list)
    assert isinstance(y, list)
    assert len(x) == len(y)


def test__load_zip_roi():
    imagej_zip_example = "tests/data/imagej_roiset.zip"

    rois = _load_zip_roi(imagej_zip_example)

    assert isinstance(rois, list)

    for (x, y) in rois:
        assert isinstance(x, list)
        assert isinstance(y, list)
        assert len(x) == len(y)


def test_roi_loaders():

    csv_example = "tests/data/roi.csv"
    imagej_roi_example = "tests/data/imagej_roi.roi"
    imagej_zip_example = "tests/data/imagej_roiset.zip"

    a = roi_loaders(csv_example)
    b = roi_loaders(imagej_roi_example)
    c = roi_loaders(imagej_zip_example)

    (x, y) = a
    assert isinstance(x, list)
    assert isinstance(y, list)
    assert len(x) == len(y)

    (x, y) = b
    assert isinstance(x, list)
    assert isinstance(y, list)
    assert len(x) == len(y)

    rois = c
    assert isinstance(rois, list)
    for (x, y) in rois:
        assert isinstance(x, list)
        assert isinstance(y, list)
        assert len(x) == len(y)
