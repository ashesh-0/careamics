import numpy as np
import pytest

from careamics.transforms.pixel_manipulation import (
    uniform_manipulate,
    _get_stratified_coords,
    _apply_struct_mask,
    median_manipulate,
    _get_indices_matrix
)


@pytest.mark.parametrize(
    "mask_pixel_perc, shape, num_iterations",
    [(0.4, (32, 32), 1000), (0.4, (10, 10, 10), 1000)],
)
def test_get_stratified_coords(mask_pixel_perc, shape, num_iterations):
    """Test the get_stratified_coords function.

    Ensure that the array of coordinates is randomly distributed across the
    image and doesn't demonstrate any strong pattern.
    """
    # Define the dummy array
    array = np.zeros(shape)

    # Iterate over the number of iterations and add the coordinates. This is an MC
    # simulation to ensure that the coordinates are randomly distributed and not
    # biased towards any particular region.
    for _ in range(num_iterations):
        # Get the coordinates of the pixels to be masked
        coords = _get_stratified_coords(mask_pixel_perc, shape)
        # Check every pair in the array of coordinates
        for coord_pair in coords:
            # Check that the coordinates are of the same shape as the patch
            assert len(coord_pair) == len(shape)
            # Check that the coordinates are positive values
            assert all(coord_pair) >= 0
            # Check that the coordinates are within the shape of the array
            assert [c <= s for c, s in zip(coord_pair, shape)]

        # Add the 1 to the every coordinate location.
        array[tuple(np.array(coords).T.tolist())] += 1

    # Ensure that there's no strong pattern in the array and sufficient number of
    # pixels is masked.
    assert np.sum(array == 0) < np.sum(shape)


def test_default_manipulate_2d(array_2D: np.ndarray):
    """Test the default_manipulate function.

    Ensure that the function returns an array of the same shape as the input.
    """
    # Get manipulated patch, original patch and mask
    patch, original_patch, mask = uniform_manipulate(array_2D, 0.5)

    # Add sample dimension to the moch input array
    array_2D = array_2D[np.newaxis, ...]
    # Check that the shapes of the arrays are the same
    assert patch.shape == array_2D.shape
    assert original_patch.shape == array_2D.shape
    assert mask.shape == array_2D.shape

    # Check that the manipulated patch is different from the original patch
    assert not np.array_equal(patch, original_patch)


def test_default_manipulate_3d(array_3D: np.ndarray):
    """Test the default_manipulate function.

    Ensure that the function returns an array of the same shape as the input.
    """
    # Get manipulated patch, original patch and mask
    patch, original_patch, mask = uniform_manipulate(array_3D, 0.5)

    # Add sample dimension to the mock input array
    array_3D = array_3D[np.newaxis, ...]
    # Check that the shapes of the arrays are the same
    assert patch.shape == array_3D.shape
    assert original_patch.shape == array_3D.shape
    assert mask.shape == array_3D.shape

    # Check that the manipulated patch is different from the original patch
    assert not np.array_equal(patch, original_patch)


# TODO what is this testing?
@pytest.mark.parametrize("mask", [[[0, 1, 1, 1, 1, 1, 0]]])
def test_apply_struct_mask(mask):
    patch = np.zeros((64, 64))
    coords = _get_stratified_coords(0.2, patch.shape)
    patch = _apply_struct_mask(patch, coords, mask)


# TODO come up with better tests for that one
@pytest.mark.parametrize("shape", 
    [
        (8, 8),
        (8, 8, 8)
    ]
)
def test_median_manipulate_ordered(ordered_array, shape):
    array = ordered_array(shape)
    patch, original_patch, mask = median_manipulate(array, 5, 0.5)

    # masked array has at least one masked pixel
    assert np.any(mask)

    # check that the arrays are different
    assert not np.array_equal(patch, original_patch)




def test_get_indices_matrix_2d():
    
    image = np.arange(15*15).reshape((15, 15))
    
    patch_size = (3, 5)
    coords = np.array(
        [
            [2, 6],
            [14, 5],
            [0, 13]
        ]
    )

    expected_result = np.array(
        [
            # first patch
            [
                [19, 20, 21, 22, 23],
                [34, 35, 36, 37, 38],
                [49, 50, 51, 52, 53]
            ],
            # second patch
            [
                [198, 199, 200, 201, 202],
                [213, 214, 215, 216, 217],
                [213, 214, 215, 216, 217]
            ],
            # third patch
            [
                [11, 12, 13, 14, 14],
                [11, 12, 13, 14, 14],
                [26, 27, 28, 29, 29]
            ]
        ]
    )

    result = _get_indices_matrix(image.shape, coords, patch_size)

    res = image[result[0], result[1]]
    assert (res == expected_result).all()


def test_get_indices_matrix_3d():
    image = np.arange(10*10*5).reshape((10, 10, 5))
    
    patch_size = (3, 5, 3)
    coords = np.array(
        [
            [2, 6, 2],
            [7, 3, 0]
        ]
    )

    expected_result = np.array(
        [
            # first patch
            [
                [
                    [ 71,  72,  73],
                    [ 76,  77,  78],
                    [ 81,  82,  83],
                    [ 86,  87,  88],
                    [ 91,  92,  93]
                ],
                [
                    [121, 122, 123],
                    [126, 127, 128],
                    [131, 132, 133],
                    [136, 137, 138],
                    [141, 142, 143]
                ],
                [
                    [171, 172, 173],
                    [176, 177, 178],
                    [181, 182, 183],
                    [186, 187, 188],
                    [191, 192, 193]
                ]
            ],
            # second patch
            [
                [
                    [305, 305, 306],
                    [310, 310, 311],
                    [315, 315, 316],
                    [320, 320, 321],
                    [325, 325, 326]
                ],
                [
                    [355, 355, 356],
                    [360, 360, 361],
                    [365, 365, 366],
                    [370, 370, 371],
                    [375, 375, 376]
                ],
                [
                    [405, 405, 406],
                    [410, 410, 411],
                    [415, 415, 416],
                    [420, 420, 421],
                    [425, 425, 426]
                ]
            ]
        ]
    )

    result = _get_indices_matrix(image.shape, coords, patch_size)

    res = image[result[0], result[1], result[2]]
    assert (res == expected_result).all()