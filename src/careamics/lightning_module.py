from typing import Any, Optional, Union

import pytorch_lightning as L
from torch import nn, optim

from careamics.config import AlgorithmModel
from careamics.config.support import (
    SupportedAlgorithm,
    SupportedArchitecture,
    SupportedLoss,
    SupportedOptimizer,
    SupportedScheduler,
)
from careamics.losses import create_loss_function
from careamics.models.model_factory import model_registry


class CAREamicsKiln(L.LightningModule):
    def __init__(self, algorithm_config: AlgorithmModel) -> None:
        super().__init__()

        # create model and loss function
        self.model: nn.Module = model_registry(algorithm_config.model)
        self.loss_func = create_loss_function(algorithm_config.loss)

        # save optimizer and lr_scheduler names and parameters
        self.optimizer_name = algorithm_config.optimizer.name
        self.optimizer_params = algorithm_config.optimizer.parameters
        self.lr_scheduler_name = algorithm_config.lr_scheduler.name
        self.lr_scheduler_params = algorithm_config.lr_scheduler.parameters

    def forward(self, x: Any) -> Any:
        return self.model(x)

    def training_step(self, batch, batch_idx) -> Any:
        x, *aux = batch
        out = self.model(x)
        loss = self.loss_func(out, *aux)
        return loss

    def validation_step(self, batch, batch_idx):
        x, *aux = batch
        out = self.model(x)
        val_loss = self.loss_func(out, *aux)

        # log validation loss
        self.log("val_loss", val_loss)

    def predict_step(self, batch, batch_idx) -> Any:
        x, *aux = batch
        out = self.model(x)
        return out, aux

    def configure_optimizers(self) -> Any:
        # instantiate optimizer
        optimizer_func = getattr(optim, self.optimizer_name)
        optimizer = optimizer_func(self.model.parameters(), **self.optimizer_params)

        # and scheduler
        scheduler_func = getattr(optim.lr_scheduler, self.lr_scheduler_name)
        scheduler = scheduler_func(optimizer, **self.lr_scheduler_params)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",  # otherwise one gets a MisconfigurationException
        }


class CAREamicsModule(CAREamicsKiln):
    def __init__(
        self,
        algorithm: Union[SupportedAlgorithm, str],
        loss: Union[SupportedLoss, str],
        architecture: Union[SupportedArchitecture, str],
        model_parameters: Optional[dict] = None,
        optimizer: Union[SupportedOptimizer, str] = "Adam",
        optimizer_parameters: Optional[dict] = None,
        lr_scheduler: Union[SupportedScheduler, str] = "ReduceLROnPlateau",
        lr_scheduler_parameters: Optional[dict] = None,
    ) -> None:
        if lr_scheduler_parameters is None:
            lr_scheduler_parameters = {}
        if optimizer_parameters is None:
            optimizer_parameters = {}
        if model_parameters is None:
            model_parameters = {}
        algorithm_configuration = {
            "algorithm": algorithm,
            "loss": loss,
            "model": {"architecture": architecture},
            "optimizer": {
                "name": optimizer,
                "parameters": optimizer_parameters,
            },
            "lr_scheduler": {
                "name": lr_scheduler,
                "parameters": lr_scheduler_parameters,
            },
        }

        # add model parameters
        algorithm_configuration["model"].update(model_parameters)

        super().__init__(AlgorithmModel(**algorithm_configuration))
