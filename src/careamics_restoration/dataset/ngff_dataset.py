from pathlib import Path
from typing import Callable, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import torch
import zarr

from careamics_restoration.config.training import ExtractionStrategies
from careamics_restoration.dataset.tiling import (
    extract_patches_predict,
    extract_patches_random,
    extract_patches_sequential,
)
from careamics_restoration.utils import normalize

from ..utils import check_axes_validity
from ..utils.logging import get_logger

logger = get_logger(__name__)


class NGFFDataset(torch.utils.data.IterableDataset):
    """Dataset to extract patches from a list of images and apply transforms to the patches."""

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
        """
        Initialize the dataset.

        Parameters
        ----------
        data_path : str
            Path to data, must be a directory.
        data_reader : Callable
            Function that reads the image data from the file. Returns an iterable of image filenames.
        patch_size : Tuple[int]
            The size of the patch to extract from the image. Must be a tuple of len either 2 or 3 depending on number of spatial dimension in the data.
        patch_generator : Union[np.ndarray, Callable]
            Function that converts an input image (item from dataset) into a iterable of image patches.
            `patch_iter(dataset[idx])` must yield a tuple: (patches, coordinates).
        image_level_transform : Optional[Callable], optional
            _description_, by default None
        patch_level_transform : Optional[Callable], optional
            _description_, by default None
        """
        self.data_path = data_path
        self.data_format = data_format
        self.axes = axes

        self.patch_transform = patch_transform

        self.files = self.list_files()

        self.mean = mean
        self.std = std
        # if not mean or not std:
        #     self.mean, self.std = self.calculate_mean_and_std()

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
        """
        Read data source and correct dimensions.

        Parameters
        ----------
        data_source : str
            Path to data source

        add_channel : bool
            If True, add channel dimension to data source

        Returns
        -------
        image volume : np.ndarray
        """
        zarr_source = zarr.open(Path(file_path), mode="r")
        if isinstance(zarr_source, zarr.hierarchy.Group):
            # get members
            pass
        elif isinstance(zarr_source, zarr.storage.DirectoryStore):
            # TODO add support for different types of storages
            pass

        elif isinstance(zarr_source, zarr.core.Array):
            # array should be of shape (S, (C), (Z), Y, X), iterating over S ?
            # TODO what if array is not of that shape and/or chunks aren't defined and
            if zarr_source.dtype == "O":
                pass
            else:
                arr = zarr_source
        else:
            raise ValueError(f"Unsupported zarr object type {type(zarr_source)}")

        # TODO how to fix dimensions? Or just raise error?
        # sanity check on dimensions
        if len(arr.shape) < 2 or len(arr.shape) > 4:
            raise ValueError(
                f"Incorrect data dimensions. Must be 2, 3 or 4 (got {arr.shape})."
            )

        # sanity check on axes length
        if len(self.axes) != len(arr.shape):
            raise ValueError(f"Incorrect axes length (got {self.axes}).")

        # check axes validity
        check_axes_validity(self.axes)  # this raises errors

        if ("S" in self.axes or "T" in self.axes) and arr.dtype != "O":
        # TODO, make sure array shape is correct
            pass
        elif arr.dtype == "O":
            #TODO check how to handle this
            for i in range(len(arr)):
                arr[i] = np.expand_dims(arr[i], axis=0)
        else:
            pass

        return arr

    def calculate_mean_and_std(self) -> Tuple[float, float]:
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
        patches = None

        if self.patch_extraction_method == ExtractionStrategies.TILED:
            patches = extract_patches_predict(
                sample, patch_size=self.patch_size, overlaps=self.patch_overlap
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
            # TODO, here sample is zarr. without occupying memory
            if self.patch_extraction_method:
                # TODO: move S and T unpacking logic from patch generator
                patches = self.generate_patches(sample)

                for patch in patches:
                    # TODO: remove this ugly workaround for normalizing 'prediction' patches
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
                # if S or T dims are not empty - assume every image is a separate sample in dim 0
                # TODO: is there always mean and std?
                for item in sample[0]:
                    item = np.expand_dims(item, (0, 1))
                    item = normalize(item, self.mean, self.std)
                    yield item
