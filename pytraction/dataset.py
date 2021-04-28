
import io
import os 
import h5py
import numpy as np
import pandas as pd 
from typing import IO, Union


class Dataset(object):
    """The Dataset class acts as a wrapper for a HDF5 bytes stream 
        or file to have pd.DataFrame like behavior. Only a single 
        column can be extracted at a time to improved run times.
    """

    def __init__(self, results: Union[str, IO[bytes]]):
        """[summary]

        Args:
            results (Union[str, IO[bytes]]): [description]
        """

        if isinstance(results, str):
            results = io.BytesIO(self.load(results))
        self.results = results
        self.columns = self._columns()

    def __str__(self):
        df = self.__getitem__(0)
        return df.__str__()

    def __repr__(self):
        df = self.__getitem__(0)
        return df.__str__()

    def __len__(self):
        with h5py.File(self.results) as f:
            length = list(f['frame'].keys())
        return len(length)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            if idx > self.__len__():
                msg = 'index out of range'
                raise IndexError(msg)
            with h5py.File(self.results) as f:
                row = {x:[np.array(f[f'{x}/{idx}'])] for x in f.keys() if 'metadata' not in x}
            return pd.DataFrame(row)
        elif isinstance(idx, str):
            with h5py.File(self.results) as f:
                items = {f'{idx}':[]}
                for i in range(self.__len__()):
                    items[idx].append(np.array(f[f'{idx}/{i}']))
            return pd.DataFrame(items)

    def _columns(self):
        return self.__getitem__(0).columns

    
    def metadata(self) -> dict:
        """Reads metadata from HDF5 bytes stream or file and returns as a dictionary.

        Returns:
            dict: Dictionary containing metadata
        """
        with h5py.File(self.results) as f:
            metadata = {x:f['metadata'].attrs[x].tostring() for x in f['metadata'].attrs.keys()}
        return metadata

    def save(self, filename:str) -> bool:
        """Saves current Dataset object as HDF5 file. 

        Args:
            filename (str): Target path to save results as HDF5.

        Returns:
            bool: True if file saved else False.
        """
        with open(filename, 'wb') as f:
            f.write(self.results.getvalue())
        if os.path.exists(filename):
            return True
        else:
            return False

    def load(self, filename):
        with open(filename, 'rb') as f:
            results = f.read()
        return results

            
        

