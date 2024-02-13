from inspect import getmembers, isclass

from enum import Enum
import albumentations as Aug

from  careamics import transforms


ALL_TRANSFORMS = dict(getmembers(Aug, isclass) + getmembers(transforms, isclass))


def get_all_transforms() -> dict:
    """Return all the transforms accepted by CAREamics.

    This includes all transforms from Albumentations (see https://albumentations.ai/),
    and custom transforms implemented in CAREamics.

    Note that while any Albumentations transform can be used in CAREamics, no check are
    implemented to verify the compatibility of any other transforms than the ones 
    officially supported (see SupportedTransforms). 

    Returns
    -------
    dict
        A dictionary with all the transforms accepted by CAREamics, where the keys are
        the transform names and the values are the transform classes.
    """
    return ALL_TRANSFORMS


class SupportedTransform(str, Enum):
    """Transforms officially supported by CAREamics.

    - Flip: from Albumentations, randomly flip the input horizontally, vertically or 
        both, parameter `p` can be used to set the probability to apply the transform.
    - RandomRotate90: from Albumentations, andomly rotate the input by 90 degrees, 
        parameter `p` can be used to set the probability to apply the transform.
    - NormalizeWithoutTarget # TODO add details, in particular about the parameters
    - ManipulateN2V # TODO add details, in particular about the parameters

    Note that while any Albumentations (see https://albumentations.ai/) transform can be
    used in CAREamics, no check are implemented to verify the compatibility of any other
    transforms than the ones officially supported. 
    """

    FLIP = "Flip"
    RANDOM_ROTATE90 = "RandomRotate90"
    NORMALIZE_WO_TARGET = "NormalizeWithoutTarget"
    MANIPULATE_N2V = "ManipulateN2V"
    # CUSTOM = "Custom"