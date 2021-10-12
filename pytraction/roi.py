import os
import zipfile
from typing import Tuple

import pandas as pd
from read_roi import read_roi_file


def _recursive_lookup(k: str, d: dict) -> list:
    """Given a nested dictionary d, return the first instance of d[k].

    Args:
        k (str): target key
        d (dict): nested dictionary

    Returns:
        list:
    """
    if k in d:
        return d[k]
    for v in d.values():
        if isinstance(v, dict):
            a = _recursive_lookup(k, v)
            if a is not None:
                return a
    return None


def _load_csv_roi(roi_path: str) -> Tuple[list, list]:
    x, y = pd.read_csv(roi_path).T.values
    return (list(x), list(y))


def _load_roireader_roi(roi_path: str) -> Tuple[list, list]:
    d = read_roi_file(roi_path)
    x = _recursive_lookup("x", d)
    y = _recursive_lookup("y", d)

    return (x, y)


def _load_zip_roi(roi_path: str) -> list:
    rois = []
    with zipfile.ZipFile(roi_path) as ziproi:
        for file in ziproi.namelist():
            roi_path_file = ziproi.extract(file)
            d = read_roi_file(roi_path_file)
            x = _recursive_lookup("x", d)
            y = _recursive_lookup("y", d)
            rois.append((x, y))
            os.remove(roi_path_file)
    return rois


def roi_loaders(roi_path):
    if ".csv" in roi_path:
        return _load_csv_roi(roi_path)

    elif ".roi" in roi_path:
        return _load_roireader_roi(roi_path)

    elif ".zip" in roi_path:
        return _load_zip_roi(roi_path)

    elif "" == roi_path:
        return None

    else:
        msg = "roi loader expecting '.csv', '.roi', or '.zip'"
        raise NotImplementedError(msg)
