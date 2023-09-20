from pathlib import Path
from typing import Callable, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import torch

from careamics_restoration.config import Configuration
from careamics_restoration.config.training import ExtractionStrategies
from careamics_restoration.dataset.tiling import (
    extract_patches_random,
    extract_patches_sequential,
    extract_tiles,
)
from careamics_restoration.dataset.dataset_utils import read_tiff, fix_axes
from careamics_restoration.manipulation import default_manipulate
from careamics_restoration.utils import normalize
from careamics_restoration.utils.logging import get_logger

logger = get_logger(__name__)


class InMemoryPathDataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str) -> None:

        self.data_path = Path(data_path)
        if not self.data_path.is_dir():
            raise ValueError("Path to data should be an existing folder.")