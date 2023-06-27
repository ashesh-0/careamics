from typing import Callable, Dict

from ..config import Configuration
from ..config.algorithm import LossName
from .losses import n2v_loss


def create_loss_function(config: Configuration) -> Callable:
    """Builds a model based on the model_name or load a checkpoint.

    _extended_summary_

    Parameters
    ----------
    model_name : _type_
        _description_
    """
    # currently the configuration accepts a list of losses or a single name
    # TODO: it is dubious whether the configuration should accept it at all
    # therefore this should simplify as soon as a decision is taken in the
    # configuration
    loss_type = config.algorithm.loss

    if len(loss_type) > 1:
        raise NotImplementedError("Multiple losses are not supported yet.")

    if loss_type[0] == LossName.n2v:
        return LossName.n2v
    elif loss_type[0] == LossName.pn2v:
        return LossName.pn2v
    else:
        raise NotImplementedError(f"Unknown loss ({loss_type[0]}).")
