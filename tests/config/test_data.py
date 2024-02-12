import pytest

from careamics.config.data import DataModel


@pytest.mark.parametrize("ext", ["nd2", "jpg", "png ", "zarr", "npy"])
def test_wrong_extensions(minimum_data: dict, ext: str):
    """Test that supported model raises ValueError for unsupported extensions."""
    minimum_data["data_type"] = ext

    # instantiate DataModel model
    with pytest.raises(ValueError):
        DataModel(**minimum_data)


@pytest.mark.parametrize("mean, std", [(0, 124.5), (12.6, 0.1)])
def test_mean_std_non_negative(minimum_data: dict, mean, std):
    """Test that non negative mean and std are accepted."""
    minimum_data["mean"] = mean
    minimum_data["std"] = std

    data_model = DataModel(**minimum_data)
    assert data_model.mean == mean
    assert data_model.std == std


def test_mean_std_negative(minimum_data: dict):
    """Test that negative mean and std are not accepted."""
    minimum_data["mean"] = -1
    minimum_data["std"] = 10.4

    with pytest.raises(ValueError):
        DataModel(**minimum_data)

    minimum_data["mean"] = 10.4
    minimum_data["std"] = -1

    with pytest.raises(ValueError):
        DataModel(**minimum_data)


def test_mean_std_both_specified_or_none(minimum_data: dict):
    """Test an error is raised if std is specified but mean is None."""
    # No error if both are None
    DataModel(**minimum_data)

    # No error if mean is defined
    minimum_data["mean"] = 10.4
    DataModel(**minimum_data)

    # Error if only std is defined
    minimum_data.pop("mean")
    minimum_data["std"] = 10.4

    with pytest.raises(ValueError):
        DataModel(**minimum_data)


def test_set_mean_and_std(minimum_data: dict):
    """Test that mean and std can be set after initialization."""
    # they can be set both, when they are already set
    data = DataModel(**minimum_data)
    data.set_mean_and_std(4.07, 14.07)

    # and if they are both None
    minimum_data["mean"] = 10.4
    minimum_data["std"] = 3.2
    data = DataModel(**minimum_data)
    data.set_mean_and_std(10.4, 0.5)


def test_patch_size(minimum_data: dict):
    """Test that non-zero even patch size are accepted."""
    minimum_data["patch_size"] = [12, 12, 12]

    data_model = DataModel(**minimum_data)
    assert data_model.patch_size == [12, 12, 12]


@pytest.mark.parametrize("patch_size",
    [
        [12],
        [0, 12, 12],
        [12, 12, 13],
        [12, 12, 12, 12]
    ]
)
def test_wrong_patch_size(minimum_data: dict, patch_size):
    """Test that wrong patch sizes are not accepted (zero or odd, dims 1 or > 3)."""
    minimum_data["patch_size"] = patch_size

    with pytest.raises(ValueError):
        DataModel(**minimum_data)


# TODO transforms validation tests
