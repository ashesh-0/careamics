from typing import List

import tifffile


class StorageDataset:
    """Dataset class for locally stored files."""

    def __init__(self) -> None:
        """Initialize the dataset."""
        self.is3d = False

    @staticmethod
    def from_memory():
        """Create dataset from objects stored in memory.

        >>> #Example
        """
        # return get_item dataset
        pass

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
