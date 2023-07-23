import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from careamics_restoration.utils.logging import ProgressLogger, get_logger

from .config import Configuration, load_configuration
from .dataset.create_dataset import (
    get_prediction_dataset,
    get_train_dataset,
    get_validation_dataset,
)
from .losses import create_loss_function
from .metrics import MetricTracker
from .models import create_model
from .prediction_utils import stitch_prediction
from .utils import (
    denormalize,
    get_device,
    normalize,
    setup_cudnn_reproducibility,
)


def seed_everything(seed: int):
    """Seed all random number generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


# TODO: discuss normalization strategies, test running mean and std
class Engine:
    """Class allowing training and prediction of a model.

    There are three ways to instantiate an Engine:
    1. With a configuration object
    2. With a configuration file, by passing a path

    In each case, the parameter name must be provided explicitly. For example:
    ``` python
    engine = Engine(config_path="path/to/config.yaml")
    ```

    Note that only one of these options can be used at a time, otherwise only one
    of them will be used, in the order of the list above.

    Parameters
    ----------
    config : Optional[Configuration], optional
        Configuration object, by default None
    config_path : Optional[Union[str, Path]], optional
        Path to configuration file, by default None
    """

    def __init__(
        self,
        *,
        config: Optional[Configuration] = None,
        config_path: Optional[Union[str, Path]] = None,
    ) -> None:
        # Sanity checks
        if config is None and config_path is None:
            raise ValueError("No configuration or path provided.")

        if config is not None:
            self.cfg = config
        elif config_path is not None:
            self.cfg = load_configuration(config_path)

        # set logging
        log_path = self.cfg.working_directory / "log.txt"
        self.progress = ProgressLogger()
        self.logger = get_logger(__name__, log_path=log_path)

        # create model, optimizer, lr scheduler and gradient scaler
        (
            self.model,
            self.optimizer,
            self.lr_scheduler,
            self.scaler,
            self.cfg,
        ) = create_model(self.cfg)
        # create loss function
        self.loss_func = create_loss_function(self.cfg)

        # use wandb or not
        if self.cfg.training is not None:
            self.use_wandb = self.cfg.training.use_wandb
        else:
            self.use_wandb = False

        if self.use_wandb:
            try:
                from wandb.errors import UsageError

                from careamics_restoration.utils.wandb import WandBLogging

                try:
                    self.wandb = WandBLogging(
                        experiment_name=self.cfg.experiment_name,
                        log_path=self.cfg.working_directory,
                        config=self.cfg,
                        model_to_watch=self.model,
                    )
                except UsageError as e:
                    self.logger.warning(
                        f"Wandb usage error, using default logger. Check whether wandb "
                        f"correctly configured:\n"
                        f"{e}"
                    )
                    self.use_wandb = False

            except ModuleNotFoundError:
                self.logger.warning(
                    "Wandb not installed, using default logger. Try pip install wandb"
                )
                self.use_wandb = False

        # get GPU or CPU device
        self.device = get_device()

        # seeding
        setup_cudnn_reproducibility(deterministic=True, benchmark=False)
        seed_everything(seed=42)

    def train(self):
        """Main train method.

        Performs training and validation steps for the specified number of epochs.

        """
        if self.cfg.training is not None:
            # General func
            train_loader = self.get_train_dataloader()

            # Set mean and std from train dataset of none
            if self.cfg.data.mean is None or self.cfg.data.std is None:
                self.cfg.data.set_mean_and_std(
                    train_loader.dataset.mean, train_loader.dataset.std
                )

            eval_loader = self.get_val_dataloader()

            self.logger.info(
                f"Starting training for {self.cfg.training.num_epochs} epochs"
            )

            val_losses = []
            try:
                for epoch in self.progress(
                    range(self.cfg.training.num_epochs),
                    task_name="Epochs",
                    overall_progress=True,
                ):  # loop over the dataset multiple times
                    train_outputs = self._train_single_epoch(
                        train_loader,
                        self.cfg.training.amp.use,
                    )

                    # Perform validation step
                    eval_outputs = self.evaluate(eval_loader)
                    self.logger.info(
                        f'Validation loss for epoch {epoch}: {eval_outputs["loss"]}'
                    )
                    # Add update scheduler rule based on type
                    self.lr_scheduler.step(eval_outputs["loss"])
                    val_losses.append(eval_outputs["loss"])
                    name = self.save_checkpoint(epoch, val_losses, "state_dict")
                    self.logger.info(f"Saved checkpoint to {name}")

                    if self.use_wandb:
                        learning_rate = self.optimizer.param_groups[0]["lr"]
                        metrics = {
                            "train": train_outputs,
                            "eval": eval_outputs,
                            "lr": learning_rate,
                        }
                        self.wandb.log_metrics(metrics)

            except KeyboardInterrupt:
                self.logger.info("Training interrupted")
                self.progress.exit()
        else:
            # TODO: instead of error, maybe fail gracefully with a logging/warning
            raise ValueError("Missing training entry in configuration file.")

    def _train_single_epoch(
        self,
        loader: torch.utils.data.DataLoader,
        amp: bool,
    ):
        """Runs a single epoch of training.

        Parameters
        ----------
        loader : torch.utils.data.DataLoader
            dataloader object for training stage
        optimizer : torch.optim.Optimizer
            optimizer object
        scaler : torch.cuda.amp.GradScaler
            scaler object for mixed precision training
        amp : bool
            whether to use automatic mixed precision
        """
        # TODO looging error LiveError: Only one live display may be active at once

        avg_loss = MetricTracker()
        self.model.to(self.device)
        self.model.train()

        for batch, *auxillary in self.progress(loader, task_name="train"):
            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=amp):
                outputs = self.model(batch.to(self.device))

            loss = self.loss_func(outputs, *auxillary, self.device)
            self.scaler.scale(loss).backward()

            avg_loss.update(loss.item(), batch.shape[0])

            self.optimizer.step()

        return {"loss": avg_loss.avg}

    def evaluate(self, eval_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Perform evaluation on the validation set.

        Parameters
        ----------
        eval_loader : torch.utils.data.DataLoader
            dataloader object for validation set

        Returns
        -------
        metrics: Dict
            validation metrics
        """
        self.model.eval()
        avg_loss = MetricTracker()

        with torch.no_grad():
            for patch, *auxillary in self.progress(
                eval_loader, task_name="validate", persistent=False
            ):
                outputs = self.model(patch.to(self.device))
                loss = self.loss_func(outputs, *auxillary, self.device)
                avg_loss.update(loss.item(), patch.shape[0])

        return {"loss": avg_loss.avg}

    def predict(
        self,
        external_input: Optional[np.ndarray] = None,
        mean: Optional[float] = None,
        std: Optional[float] = None,
    ) -> np.ndarray:
        """Inference.

        Can be used with external input or with the dataset, provided in the
        configuration file.

        Parameters
        ----------
        external_input : Optional[np.ndarray], optional
            external image array to predict on, by default None
        mean : float, optional
            mean of the train dataset, by default None
        std : float, optional
            standard deviation of the train dataset, by default None

        Returns
        -------
        np.ndarray
            predicted image array of the same shape as the input
        """
        self.model.to(self.device)
        self.model.eval()
        # TODO external input shape should either be compatible with the model or tiled.
        #  Add checks and raise errors
        if not mean and not std:
            mean = self.cfg.data.mean
            std = self.cfg.data.std

        if not mean or not std:
            raise ValueError(
                "Mean or std are not specified in the configuration and in parameters"
            )

        pred_loader, stitch = self.get_predict_dataloader(
            external_input=external_input,
            mean=mean,
            std=std,
        )
        # TODO keep getting this ValueError: Mean or std are not specified in the
        # configuration and in parameters
        # TODO where is this error? is this linked to an issue? Mention issue here.

        tiles = []
        prediction = []
        if external_input is not None:
            self.logger.info("Starting prediction on external input")
        if stitch:
            self.logger.info("Starting tiled prediction")
        else:
            self.logger.info("Starting prediction on whole sample")

        # TODO Joran/Vera: make this as a config object, add function to assess the
        # external input
        # TODO instruction unclear
        with torch.no_grad():
            # TODO tiled prediction slow af, profile and optimize
            # TODO progress bar isn't displayed
            # TODO is this linked to an issue? Mention issue here.
            for _, (tile, *auxillary) in self.progress(
                enumerate(pred_loader), task_name="Prediction"
            ):
                if auxillary:
                    (
                        last_tile,
                        sample_shape,
                        overlap_crop_coords,
                        stitch_coords,
                    ) = auxillary

                outputs = self.model(tile.to(self.device))
                outputs = denormalize(outputs, mean, std)

                if stitch:
                    # Append tile and respective coordinates to list for stitching
                    tiles.append(
                        (
                            outputs.squeeze().cpu().numpy(),
                            [list(map(int, c)) for c in overlap_crop_coords],
                            [list(map(int, c)) for c in stitch_coords],
                        )
                    )
                    # check if sample is finished
                    if last_tile:
                        # Stitch tiles together
                        predicted_sample = stitch_prediction(tiles, sample_shape)
                        prediction.append(predicted_sample)
                else:
                    prediction.append(outputs.detach().cpu().numpy().squeeze())

        self.logger.info(f"Predicted {len(prediction)} samples")
        return np.stack(prediction)

    def get_train_dataloader(self) -> DataLoader:
        """_summary_.

        _extended_summary_

        Returns
        -------
        DataLoader
            _description_
        """
        # TODO necessary for mypy, is there a better way to enforce non-null? Should
        # the training config be optional?
        if self.cfg.training is not None:
            dataset = get_train_dataset(self.cfg)
            dataloader = DataLoader(
                dataset,
                batch_size=self.cfg.training.batch_size,
                num_workers=self.cfg.training.num_workers,
                pin_memory=True,
            )
            return dataloader

        else:
            raise ValueError("Missing training entry in configuration file.")

    def get_val_dataloader(self) -> DataLoader:
        """_summary_.

        _extended_summary_

        Returns
        -------
        DataLoader
            _description_
        """
        if self.cfg.training is not None:
            dataset = get_validation_dataset(self.cfg)
            dataloader = DataLoader(
                dataset,
                batch_size=self.cfg.training.batch_size,
                num_workers=self.cfg.training.num_workers,
                pin_memory=True,
            )
            return dataloader

        else:
            raise ValueError("Missing training entry in configuration file.")

    def get_predict_dataloader(
        self,
        external_input: Optional[np.ndarray] = None,
        mean: Optional[float] = None,
        std: Optional[float] = None,
    ) -> Tuple[DataLoader, bool]:
        """_summary_.

        _extended_summary_

        Parameters
        ----------
        external_input : Optional[np.ndarray], optional
            _description_, by default None
        mean : Optional[float], optional
            _description_, by default None
        std : Optional[float], optional
            _description_, by default None

        Returns
        -------
        Tuple[DataLoader, bool]
            _description_
        """
        # TODO mypy does not take into account "is not None", we need to find a
        # workaround
        if external_input is not None and mean is not None and std is not None:
            normalized_input = normalize(external_input, mean, std)
            normalized_input = normalized_input.astype(np.float32)
            dataset = TensorDataset(torch.from_numpy(normalized_input))
            stitch = False  # TODO can also be true
        else:
            dataset = get_prediction_dataset(self.cfg)
            stitch = (
                hasattr(dataset, "patch_extraction_method")
                and dataset.patch_extraction_method is not None
            )
        return (
            # TODO this is hardcoded for now
            DataLoader(
                dataset,
                batch_size=1,  # self.cfg.prediction.data.batch_size,
                num_workers=0,  # self.cfg.prediction.data.num_workers,
                pin_memory=True,
            ),
            stitch,
        )

    def save_checkpoint(self, epoch: int, losses: List[float], save_method: str) -> str:
        """Save the model to a checkpoint file.

        Parameters
        ----------
        epoch : int
            Last epoch.
        losses : List[float]
            List of losses.
        save_method : str
            Method to save the model. Can be 'state_dict', or jit.
        """
        if epoch == 0 or losses[-1] < min(losses):
            name = f"{self.cfg.experiment_name}_best.pth"
        else:
            name = f"{self.cfg.experiment_name}_latest.pth"
        workdir = self.cfg.working_directory
        workdir.mkdir(parents=True, exist_ok=True)
        if save_method == "state_dict":
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.lr_scheduler.state_dict(),
                "grad_scaler_state_dict": self.scaler.state_dict(),
                "loss": losses[-1],
                "config": self.cfg.model_dump(),
            }
            torch.save(checkpoint, workdir / name)

        elif save_method == "jit":
            # TODO Vera help.
            # TODO add save method check in config
            raise NotImplementedError("JIT not implemented")
        else:
            raise ValueError("Invalid save method")
        return self.cfg.working_directory.absolute() / name
