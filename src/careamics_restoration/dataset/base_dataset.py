from pathlib import Path
from typing import Generator, List, Tuple, Union

import numpy as np
import tifffile
from torch import Tensor
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    IterableDataset,
    Dataset,
    get_worker_info,
)

from careamics_restoration.config.training import ExtractionStrategies
from careamics_restoration.dataset.dataset_utils import fix_axes
from careamics_restoration.dataset.tiling import (
    extract_patches_random,
    extract_patches_sequential,
    extract_tiles,
)
from careamics_restoration.utils.logging import get_logger

logger = get_logger(__name__)


class LocalStorageDataset:
    """Dataset class for locally stored files.

    Parameters could be provided as a Configuration object, path to yaml file or as
    a separate entries into dedicated methods.

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
    def from_memory(cfg, images: Union[List[np.ndarray], List[Path]]) -> Dataset:
        """Create dataset from objects that could fit into memory.

        Collects all images into a single dataset object. Can work with both numpy
        arrays and paths to tiff files. Each array should be of shape
        (1, (C), (Z), Y, X). If sample(1st) dimension is greater than 1, preprocessing
        should be applied.

        >>> #Example
        """
        if isinstance(images[0], np.ndarray):
            # create a dataset from a list of numpy arrays
            return InMemoryArrayDataset(images, cfg.data.axes)
        elif isinstance(images[0], Path):
            datasets = []
            for image in images:
                # create a dataset from a single file
                datasets.append(InMemoryArrayDataset(image, cfg.data.axes))
            return CustomConcatDataset(datasets)
        else:
            raise ValueError("Unsupported input type.")

    @staticmethod
    def from_tiff(cfg, filenames: List[str]) -> Generator:
        """Create dataset from objects stored on local disk.

        Should be used when the patched images are too large to fit into memory.

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
        raise NotImplementedError

    @staticmethod
    def zip():
        """Zip N datasets together.

        E.g., for adding ground truth to the training dataset.
        """
        raise NotImplementedError

    def shuffle(self):
        """Shuffle the dataset.

        Can be used both of files level and pathes level.
        >>> #Example
        """
        raise NotImplementedError

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
        """Generate patches from each sample.

        >>> #Example
        """
        if hasattr(self, '__getitem__'):
            return InMemoryPatchDataset(
                dataset=self, patch_size=patch_size, strategy=strategy
            )
        else:
            raise NotImplementedError

    def tiles(
        self,
        tile_size: Union[List[int], Tuple[int]],
        tile_overlap: Union[List[int], Tuple[int]],
    ) -> Generator:
        """Generate tiles from each sample.

        >>> #Example
        """
        sample = ""
        tiles = extract_tiles(arr=sample, tile_size=tile_size, overlaps=tile_overlap)
        if tiles is None:
            raise ValueError("No tiles generated.")
        yield from tiles

    def batch(self, batch_size: int) -> DataLoader:
        """Accumulate patches/tiles/samples into batches.

        Wrapper for torch.utils.data.DataLoader.

        >>> #Example
        """
        return DataLoader(self, batch_size=batch_size, num_workers=0)


class CustomConcatDataset(ConcatDataset, LocalStorageDataset):
    """Concatenates multiple datasets."""

    def __init__(self, datasets: list) -> None:
        super().__init__(datasets)


class InMemoryArrayDataset(Dataset, LocalStorageDataset):
    """Creates dataset object from numpy arrays.

    Parameters
    ----------
    data : np.ndarray
        Array containing the image
    axes: str
        Description of axes in format STCZYX
    """

    def __init__(
        self,
        data: np.ndarray,
        axes: str,
    ) -> None:
        self.data = data
        self.axes = axes

    def __len__(self) -> int:
        """Return the number of patches in the dataset."""
        return self.data.shape[0]

    def __getitem__(self, index: int) -> Tensor:
        """Returns the sample."""
        return fix_axes(self.data[index], self.axes)


class InMemoryPatchDataset(Dataset, LocalStorageDataset):
    """Creates dataset object from patches.

    Input dataset contains single sample

    Parameters
    ----------
    dataset : Dataset
        Dataset object containing the sample
    patch_size : Tuple
        Size of the patch
    strategy : str
        Patch extraction strategy
    """

    def __init__(self, dataset, patch_size: Tuple, strategy: str) -> None:
        super().__init__()
        self.dataset = dataset
        self.patch_size = patch_size
        self.strategy = strategy

        self.data = None #Extract all patches form all samples 
    def __len__(self) -> int:
        """Return the number of patches in the dataset."""
        return len(self.files)

    def __getitem__(self, index: int) -> Tensor:
        """Returns single patch."""
        for sample in self.dataset:
            if self.strategy == ExtractionStrategies.SEQUENTIAL:
                patches = extract_patches_sequential(sample, patch_size=self.patch_size)

            elif self.strategy == ExtractionStrategies.RANDOM:
                patches = extract_patches_random(sample, patch_size=self.patch_size)

            if patches is None:
                raise ValueError("No patches generated")
            return patches


class IterablePatchDataset(IterableDataset, LocalStorageDataset):
    """Creates iterable dataset object from patches.

    Dataset should have a buffer to shuffle patches from several samples.
    """

    def __init__(self, dataset, patch_size: Tuple, strategy: str) -> None:
        super().__init__()
        self.dataset = dataset
        self.patch_size = patch_size
        self.strategy = strategy
