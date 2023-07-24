import numpy as np
import torch

from careamics_restoration.config.training import ExtractionStrategies
from careamics_restoration.dataset.tiling import (
    extract_patches_random,
    extract_patches_sequential,
    extract_tiles,
)
from careamics_restoration.utils.logging import get_logger

LOGGER = get_logger(__name__)


extration_strategies = {
    ExtractionStrategies.TILED: extract_tiles,
    ExtractionStrategies.SEQUENTIAL: extract_patches_sequential,
    ExtractionStrategies.RANDOM: extract_patches_random,
}


class PatchDataset(torch.utils.data.IterableDataset):
    """Dataset for patch-based training."""

    def __init__(
        self,
        dataset,
        patch_extraction_method=None,
        patch_extraction_kwargs=None,
        patch_transform=None,
        patch_transform_params=None,
    ):
        self.dataset = dataset

        self.patch_extraction_method = patch_extraction_method

        self.patch_size = patch_extraction_kwargs.get('patch_size', None)
        self.patch_overlap = patch_extraction_kwargs.get('overlaps', None)
        self.patch_transform = patch_transform
        self.patch_transform_params = patch_transform_params

        self.mean = dataset.mean
        self.std = dataset.std

    # TODO: remove
    def generate_patches(self, sample: np.ndarray):
        """Generate patches from a sample."""
        patches = None

        if self.patch_extraction_method == ExtractionStrategies.TILED:
            patches = extract_tiles(
                sample, patch_size=self.patch_size, overlaps=self.patch_overlap
            )

        elif self.patch_extraction_method == ExtractionStrategies.SEQUENTIAL:
            patches = extract_patches_sequential(sample, patch_size=self.patch_size)

        elif self.patch_extraction_method == ExtractionStrategies.RANDOM:
            patches = extract_patches_random(sample, patch_size=self.patch_size)

        if patches is None:
            raise ValueError("No patches generated")

        return patches

    def __iter__(self):
        """Iterate over data source and yield single patch."""
        for sample in self.dataset:
            if self.patch_extraction_method:
                # TODO: move S and T unpacking logic from patch generator
                patches = self.generate_patches(sample)
                for patch in patches:
                    if self.patch_transform is not None:
                        patch = self.patch_transform(
                            patch, **self.patch_transform_params
                        )
                    yield patch

            else:
                # treat each image in 0 dim as a separate sample
                for item in sample:
                    item = np.expand_dims(item, (0, 1))
                    if self.patch_transform is not None:
                        item = self.patch_transform(
                            item, **self.patch_transform_params
                        )
                    yield item
