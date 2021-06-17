
import io
import os 
import h5py
import numpy as np
import pandas as pd 


class Dataset(object):
    """
    Small wrapper class to make hdf5 easy t
    """

    def __init__(self, log):
        if isinstance(log, str):
            log = io.BytesIO(self.load(log))
        self.log = log
        self.columns = self._columns()

    def __str__(self):
        df = self.__getitem__(0)
        return df.__str__()

    def __repr__(self):
        df = self.__getitem__(0)
        return df.__str__()

    def __len__(self):
        with h5py.File(self.log) as f:
            length = list(f['frame'].keys())
        return len(length)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            if idx > self.__len__():
                msg = 'index out of range'
                raise IndexError(msg)
            with h5py.File(self.log) as f:
                row = {x:[np.array(f[f'{x}/{idx}'])] for x in f.keys() if 'metadata' not in x}
            return pd.DataFrame(row)
        elif isinstance(idx, str):
            with h5py.File(self.log) as f:
                items = {f'{idx}':[]}
                for i in range(self.__len__()):
                    items[idx].append(np.array(f[f'{idx}/{i}']))
            return pd.DataFrame(items)

    def _columns(self):
        return self.__getitem__(0).columns

    
    def metadata(self):
        with h5py.File(self.log) as f:
            metadata = {x:f['metadata'].attrs[x].tostring() for x in f['metadata'].attrs.keys()}
        return metadata

    def save(self, filename):
        with open(filename, 'wb') as f:
            f.write(self.log.getvalue())
        if os.path.exists(filename):
            return True
        else:
            return False

    def load(self, filename):
        with open(filename, 'rb') as f:
            log = f.read()
        return log

            
        

