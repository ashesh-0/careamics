import os
import torch
import logging
import itertools
import tifffile
import numpy as np

from functools import partial
from torch.utils.data import Dataset, IterableDataset, DataLoader
from pathlib import Path
from skimage.util import view_as_windows
from typing import Callable, List, Optional, Sequence, Union, Tuple


def calculate_tile_cropping_coords(
    tile_coords: Tuple[int],
    last_tile_coord: Tuple[int],
    overlap: Tuple[int],
    image_shape: Tuple[int],
    tile_shape: Tuple[int],
) -> Tuple[slice]:
    list_coord = []
    tile_pixel_coords = []
    # Iterate over spacial dimensions (Z, Y, X)
    for i, coord in enumerate(tile_coords):
        if coord == 0:
            list_coord.append(
                slice(0, torch.div(-overlap[i], 2, rounding_mode="floor"), None)
            )
            tile_pixel_coords.append(0)
        elif coord == last_tile_coord[i] - 1:
            list_coord.append(
                slice(torch.div(overlap[i], 2, rounding_mode="floor"), None)
            )
            tile_pixel_coords.append(
                image_shape[i]
                - (tile_shape[i] - torch.div(overlap[i], 2, rounding_mode="floor"))
            )
        else:
            list_coord.append(
                slice(
                    torch.div(overlap[i], 2, rounding_mode="floor"),
                    torch.div(-overlap[i], 2, rounding_mode="floor"),
                    None,
                )
            )
            tile_pixel_coords.append(
                coord
                * (tile_shape[i] - torch.div(overlap[i], 2, rounding_mode="floor"))
            )
    return list_coord, tile_pixel_coords


def stitch_prediction(prediction, predicted_tile, sample_num, all_tiles_shape):
    pass
