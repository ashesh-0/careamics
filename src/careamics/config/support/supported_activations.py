from enum import Enum

class SupportedActivation(str, Enum):
    """Supported activation functions.
    
    - None, no activation will be used.
    - Sigmoid
    - Softmax
    - Tanh
    - ReLU
    - LeakyReLU

    All activations are defined in PyTorch.

    See: https://pytorch.org/docs/stable/nn.html#loss-functions
    """

    NONE = "None"
    SIGMOID = "Sigmoid"
    SOFTMAX = "Softmax"
    TANH = "Tanh"
    RELU = "ReLU"
    LEAKYRELU = "LeakyReLU"
