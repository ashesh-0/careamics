from pathlib import Path
import pkg_resources
from typing import Optional

import yaml
from bioimageio.spec.model.v0_5 import (
    Version,
    DocumentationSource
)

from careamics.config import Configuration
from careamics.config.support import SupportedAlgorithm
from careamics.utils import cwd, get_careamics_home

import torch
pytorch_version = Version(torch.__version__)


def _yaml_block(yaml_str: str) -> str:
    """Return a markdown code block with a yaml string.

    Parameters
    ----------
    yaml_str : str
        YAML string

    Returns
    -------
    str
        Markdown code block with the YAML string
    """
    return f"```yaml\n{yaml_str}\n```"


def readme_factory(
        config: Configuration, 
        data_description: Optional[str] = None,
        custom_description: Optional[str] = None
    ) -> Path:
    """Create a README file for the model.

    `data_description` can be used to add more information about the content of the 
    data the model was trained on.

    `custom_description` can be used to add a custom description of the algorithm, only
    used when the algorithm is set to `custom` in the configuration.

    Parameters
    ----------
    config : Configuration
        CAREamics configuration
    data_description : Optional[str], optional
        Description of the data, by default None
    custom_description : Optional[str], optional
        Description of custom algorithm, by default None

    Returns
    -------
    Path
        Path to the README file
    """
    algorithm = config.algorithm
    training = config.training
    data = config.data

    # create file
    with cwd(get_careamics_home()):
        readme = Path("README.md")
        readme.touch()

        # algorithm pretty name
        algorithm_flavour = config.get_algorithm_flavour()
        algorithm_pretty_name = algorithm_flavour + " - CAREamics"

        description = [
            f"# {algorithm_pretty_name}\n\n"
        ]

        # algorithm description
        description.append(
            "Algorithm description:\n\n"
        )
        if algorithm.algorithm == SupportedAlgorithm.CUSTOM and \
            custom_description is not None:
            description.append(custom_description)
        else:
            description.append(config.get_algorithm_description())
        description.append("\n\n")

        # algorithm details
        careamics_version = pkg_resources.get_distribution("careamics").version
        description.append(
            f"{algorithm_flavour} was trained using CAREamics (version "
            f"{careamics_version}) with the following algorithm "
            f"parameters:\n\n"
        )
        description.append(
            _yaml_block(yaml.dump(algorithm.model_dump(exclude_none=True)))
        )
        description.append(
            "\n\n"
        )

        # data description
        description.append(
            "## Data description\n\n"
        )
        if data_description is not None:
            description.append(data_description)
            description.append(
                "\n\n"
            )

        description.append(
            f"The data was processed using the following parameters:\n\n"
        )

        description.append(
            _yaml_block(yaml.dump(data.model_dump(exclude_none=True)))
        )
        description.append(
            "\n\n"
        )

        # training description
        description.append(
            "## Training description\n\n"
        )

        description.append(
            f"The model was trained using the following parameters:\n\n"
        )

        description.append(
            _yaml_block(yaml.dump(training.model_dump(exclude_none=True)))
        )
        description.append(
            "\n\n"
        )

        # references
        reference = config.get_algorithm_references()
        if reference != "":
            description.append(
                "## References\n\n"
            )
            description.append(reference)
            description.append(
                "\n\n"
            )

        # links
        description.append(
            "## Links\n\n"
            "- [CAREamics repository](https://github.com/CAREamics/careamics)\n"
            "- [CAREamics documentation](https://careamics.github.io/latest/)\n"
        )
        
        readme.write_text(
            ''.join(description)
        )

    return readme
