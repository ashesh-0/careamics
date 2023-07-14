import logging
from pathlib import Path
from typing import List, Optional, Tuple, Callable

import torch
import numpy as np
import tifffile

from careamics_restoration.utils.logging import get_logger
from careamics_restoration.utils.normalization import normalize

LOGGER = get_logger(__name__)


class TiffDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        data_path: Path,
        axes: str,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        transforms: Optional[Callable] = None,
    ):
        self.data_path = data_path
        self.axes = axes

        self.files = self.list_files()

        self.mean = mean
        self.std = std
        if not mean or not std:
            self.mean, self.std = self.calculate_mean_and_std()

        self.transforms = transforms

    # TODO: make running mean and std sheeesh
    def calculate_mean_and_std(self) -> Tuple[float, float]:
        means, stds = 0, 0
        num_samples = 0

        for file_name in self.files:
            sample = self.read_sample(file_name)
            means += sample.mean()
            stds += np.std(sample)
            num_samples += 1

        result_mean = means / num_samples
        result_std = stds / num_samples

        LOGGER.info(f"Calculated mean and std for {num_samples} images")
        LOGGER.info(f"Mean: {result_mean}, std: {result_std}")

        return result_mean, result_std

    def list_files(self) -> List[Path]:
        files = sorted(Path(self.data_path).rglob(f"*.tif*"))
        if not len(files):
            raise ValueError(f"No files found in {self.data_path}")
        return files

    def read_sample(self, file_path: Path) -> np.ndarray:
        try:
            sample = tifffile.imread(str(file_path))
            sample = sample.astype(np.float32)
        except (ValueError, OSError) as e:
            logging.exception(
                f"Exception while reading file {file_path}: {e}, skipping"
            )
            raise e

        return sample

    def fix_shape(self, sample: np.ndarray) -> np.ndarray:
        # assume that S and T are the first two axes, stack them into one
        if "S" in self.axes and "T" in self.axes:
            sample = np.vstack(sample)
        if "S" not in self.axes and "T" not in self.axes:
            sample = np.expand_dims(sample, axis=0)
        return sample

    def __len__(self):
        return len(self.files)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        for i, file_path in enumerate(self.files):
            if i % num_workers == worker_id:
                sample = self.read_sample(file_path)
                if len(self.axes) != len(sample.shape):
                    raise ValueError(
                        f"Incorrect axes length (got {self.axes} for file {file_path})."
                    )

                sample = self.fix_shape(sample)

                if self.transforms is not None:
                    sample = self.transforms(sample)

                sample = normalize(sample, self.mean, self.std)

                yield sample
