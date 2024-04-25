import numpy as np
from nd2reader import ND2Reader

def load_one_file(fpath, axes=None):
    """
    '/group/jug/ashesh/data/pavia3_sequential/Cond_2/Main/1_002.nd2'
    """
    output = {}
    with ND2Reader(str(fpath)) as fobj:
        for c in range(len(fobj.metadata['channels'])):
            output[c] = []
            for z in fobj.metadata['z_levels']:
                img = fobj.get_frame_2D(c=c, z=z)
                img = img[None, ..., None]
                output[c].append(img)
            output[c] = np.concatenate(output[c], axis=0)
    data = np.concatenate([output[0], output[1]], axis=-1)
    return data
