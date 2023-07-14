from careamics_restoration.config import Configuration
from careamics_restoration.config.training import ExtractionStrategies
from careamics_restoration.dataset.tiff_dataset import TiffDataset
from careamics_restoration.dataset.patch_dataset import PatchDataset
from careamics_restoration.manipulation import default_manipulate


def get_train_dataset(config: Configuration) -> TiffDataset:
    """
    Create TiffDataset instance from configuration

    Yields
    ------
    TiffDataset
    """
    if config.training is None:
        raise ValueError("Training configuration is not defined.")

    data_path = config.data.training_path

    tiff_data = TiffDataset(
        data_path=data_path,
        axes=config.data.axes,
        mean=config.data.mean,
        std=config.data.std
    )

    patch_data = PatchDataset(
        dataset=tiff_data,
        patch_extraction_method=config.training.extraction_strategy,
        patch_exrtaction_kwargs={'patch_size': config.training.patch_size},
        patch_transform=default_manipulate,
        patch_transform_params={
            "mask_pixel_percentage": config.algorithm.masked_pixel_percentage
        },
    )

    return patch_data


def get_validation_dataset(config: Configuration) -> TiffDataset:
    """
    Create TiffDataset instance from configuration

    Yields
    ------
    TiffDataset
    """
    if config.training is None:
        raise ValueError("Training configuration is not defined.")

    data_path = config.data.validation_path

    tiff_data = TiffDataset(
        data_path=data_path,
        axes=config.data.axes,
        mean=config.data.mean,
        std=config.data.std
    )

    patch_data = PatchDataset(
        dataset=tiff_data,
        patch_extraction_method=config.training.extraction_strategy,
        patch_exrtaction_kwargs={'patch_size': config.training.patch_size},
        patch_transform=default_manipulate,
        patch_transform_params={
            "mask_pixel_percentage": config.algorithm.masked_pixel_percentage
        },
    )

    return patch_data


def get_prediction_dataset(config: Configuration) -> TiffDataset:
    """
    Create TiffDataset instance from configuration

    Yields
    ------
    TiffDataset
    """
    if config.prediction is None:
        raise ValueError("Prediction configuration is not defined.")

    if config.prediction.use_tiling:
        patch_extraction_method = ExtractionStrategies.TILED
    else:
        patch_extraction_method = None

    tiff_data = TiffDataset(
        data_path=config.data.prediction_path,
        axes=config.data.axes,
        mean=config.data.mean,
        std=config.data.std,
    )

    patch_data = PatchDataset(
        dataset=tiff_data,
        patch_extraction_method=patch_extraction_method,
        patch_exrtaction_kwargs={'patch_size': config.prediction.tile_shape, 'overlaps': config.prediction.overlaps},
        patch_transform=None,
        patch_transform_params=None
    )

    return patch_data
