import logging
from pathlib import Path
from typing import Callable, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import tifffile
import torch

from careamics_restoration.config.training import ExtractionStrategies
from careamics_restoration.dataset.tiling import (
    extract_patches_random,
    extract_patches_sequential,
    extract_tiles,
)
from careamics_restoration.utils import normalize
from careamics_restoration.utils.logging import get_logger

logger = get_logger(__name__)


class TiffDataset(torch.utils.data.IterableDataset):
    """Dataset to extract patches from a tiff image(s).

    Parameters
    ----------
    data_path : str
        Path to data, must be a directory.

    axes: str
        Description of axes in format STCZYX

    patch_extraction_method: ExtractionStrategies
        Patch extraction strategy, one of "sequential", "random", "tiled"

    patch_size : Tuple[int]
        The size of the patch to extract from the image. Must be a tuple of len either
        2 or 3 depending on number of spatial dimension in the data.

    patch_overlap: Tuple[int]
        Size of the overlaps. Used for "tiled" tiling strategy.

    mean: float
        Expected mean of the samples

    std: float
        Expected std of the samples

    patch_transform: Optional[Callable], optional
        Transform to apply to patches.

    patch_transform_params: Optional[Dict], optional
        Additional parameters to pass to patch transform function
    """

    def __init__(
        self,
        data_path: Union[Path, str],
        data_format: str,
        axes: str,
        patch_extraction_method: ExtractionStrategies,
        patch_size: Optional[Union[List[int], Tuple[int]]] = None,
        patch_overlap: Optional[Union[List[int], Tuple[int]]] = None,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        patch_transform: Optional[Callable] = None,
        patch_transform_params: Optional[Dict] = None,
    ) -> None:
        self.data_path = data_path
        self.data_format = data_format
        self.axes = axes

        self.patch_transform = patch_transform

        self.files = self.list_files()

        self.mean = mean
        self.std = std
        if not mean or not std:
            self.mean, self.std = self.calculate_mean_and_std()

        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.patch_extraction_method = patch_extraction_method
        self.patch_transform = patch_transform
        self.patch_transform_params = patch_transform_params

    def list_files(self) -> List[Path]:
        """Creates a list of paths to source tiff files from path string.

        Returns
        -------
        List[Path]
            List of pathlib.Path objects
        """
        files = sorted(Path(self.data_path).rglob(f"*.{self.data_format}*"))
        return files

    def read_image(self, file_path: Path) -> np.ndarray:
        """_summary_.

        Parameters
        ----------
        file_path : Path
            _description_

        Returns
        -------
        np.ndarray
            _description_

        Raises
        ------
        e
            _description_
        ValueError
            _description_
        ValueError
            _description_
        """
        if file_path.suffix == ".npy":
            try:
                sample = np.load(file_path)
            except ValueError:
                sample = np.load(file_path, allow_pickle=True)

        elif file_path.suffix[:4] == ".tif":
            try:
                sample = tifffile.imread(file_path)
            except (ValueError, OSError) as e:
                logging.exception(f"Exception in file {file_path}: {e}, skipping")
                raise e

        sample = sample.squeeze()

        if len(sample.shape) < 2 or len(sample.shape) > 4:
            raise ValueError(
                f"Incorrect data dimensions. Must be 2, 3 or 4 (got {sample.shape}"
                f"for file {file_path})."
            )

        # check number of axes
        if len(self.axes) != len(sample.shape):
            raise ValueError(
                f"Incorrect axes length (got {self.axes} for file {file_path})."
            )
        sample = self.fix_axes(sample)
        return sample

    def calculate_mean_and_std(self) -> Tuple[float, float]:
        """Calculate mean and std of the dataset."""
        means, stds = 0, 0
        num_samples = 0

        for sample in self.iterate_files():
            means += sample.mean()
            stds += np.std(sample)
            num_samples += 1

        result_mean = means / num_samples
        result_std = stds / num_samples

        logger.info(f"Calculated mean and std for {num_samples} images")
        logger.info(f"Mean: {result_mean}, std: {result_std}")
        # TODO pass stage here to be more explicit with logging
        return result_mean, result_std

    def fix_axes(self, sample: np.ndarray) -> np.ndarray:
        """Fix axes order and add missing axes."""
        # concatenate ST axes to N, return NCZYX
        if ("S" in self.axes or "T" in self.axes) and sample.dtype != "O":
            new_axes_len = len(self.axes.replace("Z", "").replace("YX", ""))
            # TODO test reshape, replace with moveaxis ?
            sample = sample.reshape(-1, *sample.shape[new_axes_len:]).astype(np.float32)

        elif sample.dtype == "O":
            for i in range(len(sample)):
                sample[i] = np.expand_dims(sample[i], axis=0).astype(np.float32)

        else:
            sample = np.expand_dims(sample, axis=0).astype(np.float32)

        return sample

    def generate_patches(self, sample: np.ndarray) -> Generator[np.ndarray, None, None]:
        """Generate patches from a sample."""
        patches = None

        if self.patch_extraction_method == ExtractionStrategies.TILED:
            patches = extract_tiles(
                sample, tile_size=self.patch_size, overlaps=self.patch_overlap
            )

        elif self.patch_extraction_method == ExtractionStrategies.SEQUENTIAL:
            patches = extract_patches_sequential(sample, patch_size=self.patch_size)

        elif self.patch_extraction_method == ExtractionStrategies.RANDOM:
            patches = extract_patches_random(sample, patch_size=self.patch_size)

        if patches is None:
            raise ValueError("No patches generated")

        return patches

    def iterate_files(self) -> np.ndarray:
        """
        Iterate over data source and yield whole image.

        Yields
        ------
        np.ndarray
        """
        # When num_workers > 0, each worker process will have a different copy of the
        # dataset object
        # Configuring each copy independently to avoid having duplicate data returned
        # from the workers
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        for i, filename in enumerate(self.files):
            if i % num_workers == worker_id:
                sample = self.read_image(filename)
                yield sample

    def __iter__(self) -> np.ndarray:
        """
        Iterate over data source and yield single patch.

        Yields
        ------
        np.ndarray
        """
        for sample in self.iterate_files():
            # TODO patch_extraction_method should never be None!
            if self.patch_extraction_method:
                # TODO: move S and T unpacking logic from patch generator
                patches = self.generate_patches(sample)

                for patch in patches:
                    # TODO: remove this ugly workaround for normalizing 'prediction'
                    #  patches
                    if isinstance(patch, tuple):
                        normalized_patch = normalize(patch[0], self.mean, self.std)
                        patch = (normalized_patch, *patch[1:])
                    else:
                        patch = normalize(patch, self.mean, self.std)

                    if self.patch_transform is not None:
                        patch = self.patch_transform(
                            patch, **self.patch_transform_params
                        )

                    yield patch

            else:
                # if S or T dims are not empty - assume every image is a separate
                # sample in dim 0
                # TODO: is there always mean and std?
                for item in sample[0]:
                    item = np.expand_dims(item, (0, 1))
                    item = normalize(item, self.mean, self.std)
                    yield item
