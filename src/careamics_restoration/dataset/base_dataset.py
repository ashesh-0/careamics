from typing import List

import numpy as np
import tifffile
from torch.utils.data import ConcatDataset, Dataset

from careamics_restoration.dataset.tensor_dataset import InMemoryDataset


class LocalStorageDataset:
    """Dataset class for locally stored files."""

    def __init__(self) -> None:
        """Initialize the dataset."""
        self.is3d = False

    @staticmethod
    def from_memory(cfg, images: List[np.ndarray]) -> List[Dataset]:
        """Create dataset from objects stored in memory.

        >>> #Example
        """
        # return get_item dataset
        datasets = []
        for image in images:
            # create a dataset from a single file
            datasets.append(InMemoryDataset(cfg, image))
        return ConcatDataset(datasets)

    @staticmethod
    def from_tiff(cfg, filenames: List[str]):
        """Create dataset from objects stored on local disk.

        >>> #Example
        """
        for filename in filenames:
            contents = tifffile.TiffFile(filename)
            for _page in contents.pages:
                if cfg.is3d:
                    # TODO
                    pass
                else:
                    pass
                # create a dataset from a single file

        # TODO could there be more than one file inside tifffile? All other metadata ?

    @staticmethod
    def from_zarr():
        """Create dataset from zarr file.

        >>> #Example
        """
        pass
