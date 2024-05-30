from czifile import imread as imread_czi
import numpy as np

def load_data(fpath, axes=None):
    data = imread_czi(fpath)
    if len(data.shape) > 4:
        # (4, 1, 4, 22, 512, 512, 1)
        clean_data = data[3, 0, [0, 2], ..., 0]
        clean_data = np.swapaxes(clean_data[..., None], 0, 4)[0]
    else:
        # just for iba dataset. (2,H,W,1) -> (2,H,W)
        clean_data = data.squeeze()

    print('Loaded from', fpath, 'shape:', clean_data.shape)
    return clean_data

