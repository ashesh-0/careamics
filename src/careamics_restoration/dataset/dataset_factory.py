from careamics_restoration.config import Configuration
from careamics_restoration.config.training import ExtractionStrategies
from careamics_restoration.dataset.ngff_dataset import NGFFDataset
from careamics_restoration.dataset.tiff_dataset import TiffDataset
from careamics_restoration.manipulation import default_manipulate


def get_train_dataset(config: Configuration) -> TiffDataset:
    """
    Create TiffDataset instance from configuration.

    Yields
    ------
    TiffDataset
    """
    if config.training is None:
        raise ValueError("Training configuration is not defined.")

    data_path = config.data.training_path
    if config.data.data_format == "zarr":
        dataset = NGFFDataset(
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
    else:
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


def get_validation_dataset(config: Configuration) -> TiffDataset:
    if config.training is None:
        raise ValueError("Training configuration is not defined.")

    data_path = config.data.validation_path

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


def get_prediction_dataset(config: Configuration) -> TiffDataset:
    if config.prediction is None:
        raise ValueError("Prediction configuration is not defined.")

    if config.prediction.use_tiling:
        patch_extraction_method = ExtractionStrategies.TILED
    else:
        patch_extraction_method = None

    dataset = TiffDataset(
        data_path=config.data.prediction_path,
        data_format=config.data.data_format,
        axes=config.data.axes,
        mean=config.data.mean,
        std=config.data.std,
        patch_size=config.prediction.tile_shape,
        patch_overlap=config.prediction.overlaps,
        patch_extraction_method=patch_extraction_method,
        patch_transform=None,
    )

    return dataset
