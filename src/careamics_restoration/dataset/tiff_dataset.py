from pathlib import Path
from typing import Callable, Dict, Generator, Iterator, List, Optional, Tuple, Union

import numpy as np
from torch.utils.data import IterableDataset

from careamics_restoration.config import Configuration
from careamics_restoration.config.training import ExtractionStrategies
from careamics_restoration.manipulation import default_manipulate
from careamics_restoration.utils import check_tiling_validity, normalize
from careamics_restoration.utils.logging import get_logger

logger = get_logger(__name__)


class ShuffleIterableDataset(IterableDataset):
    """Dataset that supports shuffling within a buffer size.

    Parameters
    ----------
    dataset : IterableDataset
    """

    def __iter__(self) -> Iterator:
        """Iterate over all samples."""
        for dataset in self.datasets:
            assert isinstance(
                dataset, IterableDataset
            ), "Only IterableDataset are supported"
            for data in dataset:
                yield data


class TiffDataset(IterableDataset):
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
        data_path: Union[str, Path],
        data_format: str,
        axes: str,
        patch_extraction_method: Union[ExtractionStrategies, None],
        patch_size: Optional[Union[List[int], Tuple[int]]] = None,
        patch_overlap: Optional[Union[List[int], Tuple[int]]] = None,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        patch_transform: Optional[Callable] = None,
        patch_transform_params: Optional[Dict] = None,
    ) -> None:
        self.data_path = Path(data_path)
        if not self.data_path.is_dir():
            raise ValueError("Path to data should be an existing folder.")

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

    def calculate_mean_and_std(self) -> Tuple[float, float]:
        """Calculates mean and std of the dataset.

        Returns
        -------
        Tuple[float, float]
            Tuple containing mean and std
        """
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

    def generate_patches(self, sample: np.ndarray) -> Generator[np.ndarray, None, None]:
        """Generate patches from a sample.

        Parameters
        ----------
        sample : np.ndarray
            array containing the image

        Yields
        ------
        Generator[np.ndarray, None, None]
            Generator function yielding patches/tiles

        Raises
        ------
        ValueError
            if no patches are generated
        """
        patches = None
        assert self.patch_size is not None, "Patch size must be provided"

        if self.patch_extraction_method == ExtractionStrategies.TILED:
            assert self.patch_overlap is not None, "Patch overlap must be provided"
            patches = extract_tiles(
                arr=sample, tile_size=self.patch_size, overlaps=self.patch_overlap
            )

        elif self.patch_extraction_method == ExtractionStrategies.SEQUENTIAL:
            patches = extract_patches_sequential(sample, patch_size=self.patch_size)

        elif self.patch_extraction_method == ExtractionStrategies.RANDOM:
            patches = extract_patches_random(sample, patch_size=self.patch_size)

        if patches is None:
            raise ValueError("No patches generated")

        return patches

    def __iter__(self) -> Generator[np.ndarray, None, None]:
        """
        Iterate over data source and yield single patch.

        Yields
        ------
        np.ndarray
        """
        assert (
            self.mean is not None and self.std is not None
        ), "Mean and std must be provided"
        for sample in self.iterate_files():
            # TODO patch_extraction_method should never be None!
            if self.patch_extraction_method:
                # TODO: move S and T unpacking logic from patch generator
                patches = self.generate_patches(sample)

                for patch in patches:
                    # TODO: remove this ugly workaround for normalizing 'prediction'
                    # patches
                    if isinstance(patch, tuple):
                        normalized_patch = normalize(
                            img=patch[0], mean=self.mean, std=self.std
                        )
                        patch = (normalized_patch, *patch[1:])
                    else:
                        patch = normalize(img=patch, mean=self.mean, std=self.std)

                    if self.patch_transform is not None:
                        assert self.patch_transform_params is not None
                        patch = self.patch_transform(
                            patch, **self.patch_transform_params
                        )

                    yield patch

            else:
                # if S or T dims are not empty - assume every image is a separate
                # sample in dim 0
                for i in range(sample.shape[0]):
                    item = np.expand_dims(sample[i], (0, 1))
                    item = normalize(img=item, mean=self.mean, std=self.std)
                    yield item


def get_train_dataset(config: Configuration, train_path: str) -> TiffDataset:
    """Create Dataset instance from configuration.

    Parameters
    ----------
    config : Configuration
        Configuration object
    train_path : Union[str, Path]
        Pathlike object with a path to training data

    Returns
    -------
        Dataset object

    Raises
    ------
    ValueError
        No training configuration found
    """
    if config.training is None:
        raise ValueError("Training configuration is not defined.")

    dataset = TiffDataset(
        data_path=train_path,
        data_format=config.data.data_format,
        axes=config.data.axes,
        mean=config.data.mean,
        std=config.data.std,
        patch_extraction_method=config.training.extraction_strategy,
        patch_size=config.training.patch_size,
        patch_transform=default_manipulate,
        patch_transform_params={
            "mask_pixel_percentage": config.algorithm.masked_pixel_percentage,
            "roi_size": config.algorithm.roi_size,
        },
    )
    return dataset


def get_validation_dataset(config: Configuration, val_path: str) -> TiffDataset:
    """Create Dataset instance from configuration.

    Parameters
    ----------
    config : Configuration
        Configuration object
    val_path : Union[str, Path]
        Pathlike object with a path to validation data

    Returns
    -------
    TiffDataset
        Dataset object

    Raises
    ------
    ValueError
        No validation configuration found
    """
    if config.training is None:
        raise ValueError("Training configuration is not defined.")

    data_path = val_path

    dataset = TiffDataset(
        data_path=data_path,
        data_format=config.data.data_format,
        axes=config.data.axes,
        mean=config.data.mean,
        std=config.data.std,
        patch_extraction_method=config.training.extraction_strategy,
        patch_size=config.training.patch_size,
        patch_transform=default_manipulate,
        patch_transform_params={
            "mask_pixel_percentage": config.algorithm.masked_pixel_percentage
        },
    )

    return dataset


def get_prediction_dataset(
    config: Configuration,
    pred_path: Union[str, Path],
    *,
    tile_shape: Optional[List[int]] = None,
    overlaps: Optional[List[int]] = None,
    axes: Optional[str] = None,
) -> TiffDataset:
    """Create Dataset instance from configuration.

    To use tiling, both `tile_shape` and `overlaps` must be specified, have same
    length, be divisible by 2 and greater than 0. Finally, the overlaps must be
    smaller than the tiles.

    Parameters
    ----------
    config : Configuration
        Configuration object
    pred_path : Union[str, Path]
        Pathlike object with a path to prediction data
    tile_shape : Optional[List[int]], optional
        2D or 3D shape of the tiles to be predicted, by default None
    overlaps : Optional[List[int]], optional
        2D or 3D overlaps between tiles, by default None
    axes : Optional[str], optional
        Axes of the data, by default None

    Returns
    -------
    TiffDataset
        Dataset object

    Raises
    ------
    ValueError

    """
    use_tiling = False  # default value

    # Validate tiles and overlaps
    if tile_shape is not None and overlaps is not None:
        check_tiling_validity(tile_shape, overlaps)

        # Use tiling
        use_tiling = True

    # Extraction strategy
    if use_tiling:
        patch_extraction_method = ExtractionStrategies.TILED
    else:
        patch_extraction_method = None

    # Create dataset
    dataset = TiffDataset(
        data_path=pred_path,
        data_format=config.data.data_format,
        axes=config.data.axes if axes is None else axes,  # supersede axes if provided
        mean=config.data.mean,
        std=config.data.std,
        patch_size=tile_shape,
        patch_overlap=overlaps,
        patch_extraction_method=patch_extraction_method,
        patch_transform=None,
    )

    return dataset
