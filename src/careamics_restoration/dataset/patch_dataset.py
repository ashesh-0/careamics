import numpy as np
import torch

from careamics_restoration.config.training import ExtractionStrategies
from careamics_restoration.dataset.tiling import (
    extract_patches_predict,
    extract_patches_sequential,
    extract_patches_random,
)
from careamics_restoration.utils.logging import get_logger

LOGGER = get_logger(__name__)


extration_strategies = {
    ExtractionStrategies.TILED: extract_patches_predict,
    ExtractionStrategies.SEQUENTIAL: extract_patches_sequential,
    ExtractionStrategies.RANDOM: extract_patches_random,
}


class PatchDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        dataset,
        patch_extraction_method=None,
        patch_exrtaction_kwargs=None,
        patch_transform=None,
        patch_transform_params=None,
    ):
        self.dataset = dataset

        self.patch_extractor = extration_strategies.get(patch_extraction_method, None)

        self.patch_exrtaction_args = patch_exrtaction_kwargs
        self.patch_transform = patch_transform
        self.patch_transform_params = patch_transform_params

    def __iter__(self):
        for sample in self.dataset:
            if self.patch_extractor:
                # TODO: move S and T unpacking logic from patch generator
                patches = self.patch_extractor(
                    sample, kwargs=self.patch_exrtaction_kwargs
                )
                for patch in patches:
                    if self.patch_transform is not None:
                        patch = self.patch_transform(
                            patch, kwargs=self.patch_transform_params
                        )
                    yield patch

            else:
                # treat each image in 0 dim as a separate sample
                for item in sample:
                    item = np.expand_dims(item, (0, 1))
                    if self.patch_transform is not None:
                        item = self.patch_transform(
                            item, kwargs=self.patch_transform_params
                        )
                    yield item
