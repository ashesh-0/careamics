from pathlib import Path
from typing import Generator, List, Tuple, Union

import numpy as np
import tifffile
from torch.utils.data import ConcatDataset, Dataset, get_worker_info

from careamics_restoration.config.training import ExtractionStrategies
from careamics_restoration.dataset.tensor_dataset import InMemoryDataset
from careamics_restoration.dataset.tiling import (
    extract_patches_random,
    extract_patches_sequential,
    extract_tiles,
)
from careamics_restoration.utils.logging import get_logger

logger = get_logger(__name__)


class LocalStorageDataset:
    """Dataset class for locally stored files.

    >>> #Example
    # dataset.from_fiff.
    """

    def __init__(self) -> None:
        """Initialize the dataset."""
        self.is3d = False

    def list_files(self) -> List[Path]:
        """Creates a list of paths to source tiff files from path string.

        Returns
        -------
        List[Path]
            List of pathlib.Path objects
        """
        files = sorted(Path(self.data_path).rglob(f"*.{self.data_format}*"))
        return files

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
    def from_tiff(cfg, filenames: List[str]) -> Generator:
        """Create dataset from objects stored on local disk.

        >>> #Example
        """
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        for i, filename in enumerate(filenames):
            if i % num_workers != worker_id:
                contents = tifffile.TiffFile(filename)
                try:
                    if cfg.is3d:
                        # for page in contents.pages:
                        pass
                    else:
                        contents.pages[0].asarray()
                except AttributeError:
                    logger.warning(f"File {filename} is not a valid tiff file.")
                    # TODO try imread
                    tifffile.imread(filename)

                # create a dataset from a single file

        # TODO could there be more than one file inside tifffile? All other metadata ?

    @staticmethod
    def from_zarr():
        """Create dataset from zarr file.

        >>> #Example
        """
        pass
    @staticmethod
    def zip():
        """Zip N datasets together.

        E.g., for adding ground truth to the training dataset.
        """
        pass

    def shuffle(self):
        """Shuffle the dataset.

        Can be used both of files level and pathes level.
        >>> #Example
        """
        pass

    def apply(self):
        """Apply transformations to the dataset.

        >>> #Example
        """
        pass

    def patch(
        self,
        patch_size: Union[List[int], Tuple[int]],
        strategy: ExtractionStrategies,
    ):
        """Generate pathes from each sample.

        >>> #Example
        """
        for sample in self:
            if strategy == ExtractionStrategies.SEQUENTIAL:
                patches = extract_patches_sequential(sample, patch_size=patch_size)

            elif strategy == ExtractionStrategies.RANDOM:
                patches = extract_patches_random(sample, patch_size=patch_size)

            if patches is None:
                raise ValueError("No patches generated")
            yield from patches

    def tiles(
        self,
        tile_size: Union[List[int], Tuple[int]],
        tile_overlap: Union[List[int], Tuple[int]],
    ) -> Generator:
        """Generate tiles from each sample.

        >>> #Example
        """
        tiles = extract_tiles(arr=sample, tile_size=tile_size, overlaps=tile_overlap)
        if tiles is None:
            raise ValueError("No tiles generated.")
        yield from tiles
