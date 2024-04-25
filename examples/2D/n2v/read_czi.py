from czifile import imread as imread_czi
import numpy as np

def load_data(fpath, axes=None):
    # (4, 1, 4, 22, 512, 512, 1)
    data = imread_czi(fpath)
    clean_data = data[3, 0, [0, 2], ..., 0]
    clean_data = np.swapaxes(clean_data[..., None], 0, 4)[0]
    print('Loaded from', fpath, 'shape:', clean_data.shape)
    return clean_data

