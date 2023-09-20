import logging
from pathlib import Path

import numpy as np
import tifffile
import torch


def fix_axes(sample: np.ndarray, axes: str) -> np.ndarray:
    """Fixes axes of the sample to match the config axes.

    Parameters
    ----------
    sample : np.ndarray
        array containing the image

    Returns
    -------
    np.ndarray
        reshaped array
    """
    # concatenate ST axes to N, return NCZYX
    if "S" in axes or "T" in axes:
        new_axes_len = len(axes.replace("Z", "").replace("YX", ""))
        sample = sample.reshape(-1, *sample.shape[new_axes_len:]).astype(np.float32)

    if ("Z" in axes and len(sample.shape) < 4) or (
        "Z" not in axes and len(sample.shape) < 3
    ):
        sample = np.expand_dims(sample, axis=0).astype(np.float32)

    return sample


def read_tiff(self, file_path: Path) -> np.ndarray:
    """Reads a file and returns a numpy array.

    Parameters
    ----------
    file_path : Path
        pathlib.Path object containing a path to a file

    Returns
    -------
    np.ndarray
        array containing the image

    Raises
    ------
    ValueError, OSError
        if a file is not a valid tiff or damaged
    ValueError
        if data dimensions are not 2, 3 or 4
    ValueError
        if axes parameter from config is not consistent with data dimensions
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
            f"Incorrect data dimensions. Must be 2, 3 or 4 (got {sample.shape} for"
            f"file {file_path})."
        )

    # check number of axes
    if len(self.axes) != len(sample.shape):
        raise ValueError(
            f"Incorrect axes length (got {self.axes} for file {file_path})."
        )
    sample = fix_axes(sample)
    return sample
