from enum import Enum

import torch
import torch.nn as nn

from typing import List
from collections import OrderedDict
from torch.nn import init

from .layers import DownConv, UpConv, conv1x1, Conv_Block_tf, BlurPool2d, BlurPool3d


# TODO add docstings, typing


class UNET(nn.Module):
    def __init__(
        self,
        conv_dim: int,
        num_classes: int = 1,
        in_channels: int = 1,
        depth: int = 3,
        num_filter_base: int = 64,
        num_conv_per_depth=2,
        activation="ReLU",
        use_batch_norm=True,
        dropout=0.0,
        pool_kernel=2,
        last_activation=None,
        n2v2: bool = False,
        skip_skipone=False,
    ) -> None:
        super().__init__()

        self.depth = depth
        self.num_conv_per_depth = num_conv_per_depth
        self.conv_block_in = Conv_Block_tf
        self.conv_block = Conv_Block_tf

        if n2v2:
            self.pooling = (
                BlurPool2d if conv_dim == 2 else BlurPool3d
            )  # TODO getattr from layers
        else:
            self.pooling = getattr(nn, f"MaxPool{conv_dim}d")(kernel_size=pool_kernel)

        self.upsampling = nn.Upsample(
            scale_factor=2, mode="bilinear"
        )  # TODO check align_corners and mode
        self.skipone = (
            skip_skipone  # TODO whoever called this skip_skipone must explain himself
        )
        enc_blocks = OrderedDict()
        bottleneck = OrderedDict()
        dec_blocks = OrderedDict()
        self.skip_layers_ouputs = OrderedDict()

        # TODO implements better layer naming
        # Encoder
        for n in range(self.depth):
            for i in range(self.num_conv_per_depth):
                in_channels = in_channels if i == 0 else out_channels
                out_channels = num_filter_base * (2**n)
                layer = self.conv_block(
                    conv_dim,
                    in_channels,
                    out_channels,
                    stride=1,
                    padding=1,
                    bias=True,
                    groups=1,
                    activation="ReLU",
                    dropout_perc=0,
                    use_batch_norm=use_batch_norm,
                )
                enc_blocks[f"encoder_conv_d{n}_num{i}"] = layer

            if skip_skipone:
                if n > 0:
                    enc_blocks[f"skip_encoder_conv_d{n}"] = enc_blocks.pop(
                        f"encoder_conv_d{n}_num{i}"
                    )
            else:
                enc_blocks[f"skip_encoder_conv_d{n}"] = enc_blocks.pop(
                    f"encoder_conv_d{n}_num{i}"
                )

            enc_blocks[f"encoder_pool_d{n}"] = self.pooling

        # Bottleneck
        for i in range(num_conv_per_depth - 1):
            bottleneck[f"bottleneck_num{i}"] = self.conv_block(
                conv_dim,
                in_channels=out_channels,
                out_channels=num_filter_base * 2**depth,
                stride=1,
                padding=1,
                bias=True,
                groups=1,
                activation="ReLU",
                dropout_perc=0,
                use_batch_norm=use_batch_norm,
            )

        bottleneck["bottleneck_final"] = self.conv_block(
            conv_dim,
            in_channels=num_filter_base * 2**depth,
            out_channels=num_filter_base * 2 ** max(0, depth - 1),
            dropout=dropout,
            activation=activation,
            use_batch_norm=use_batch_norm,
        )

        # Decoder
        for n in reversed(range(depth)):
            dec_blocks[f"upsampling_d{n}"] = self.upsampling
            for i in range(num_conv_per_depth - 1):
                n_filter = num_filter_base * 2**n if n > 0 else num_filter_base
                dec_blocks[f"decoder_conv_d{n}_num{i}"] = self.conv_block(
                    conv_dim,
                    in_channels=n_filter * 2,
                    out_channels=n_filter,
                    dropout=dropout,
                    activation=activation,
                    use_batch_norm=use_batch_norm,
                )

            dec_blocks[f"decoder_conv_d{n}"] = self.conv_block(
                conv_dim,
                in_channels=n_filter,
                out_channels=num_filter_base * 2 ** max(0, n - 1),
                dropout=dropout,
                activation=activation if n > 0 else last_activation,
                use_batch_norm=use_batch_norm,
            )

        self.enc_blocks = nn.ModuleDict(enc_blocks)
        self.bottleneck = nn.ModuleDict(bottleneck)
        self.dec_blocks = nn.ModuleDict(dec_blocks)
        self.final_conv = getattr(nn, f"Conv{conv_dim}d")(
            in_channels=num_filter_base * 2 ** max(0, n - 1),
            out_channels=num_classes,
            kernel_size=1,
        )

    def forward(self, x):
        # TODO certain input sizes lead to shape mismatch, eg 160x150
        inputs = x.clone()
        for module_name in self.enc_blocks:
            x = self.enc_blocks[module_name](x)
            if module_name.startswith("skip"):
                self.skip_layers_ouputs[module_name] = x

        for module_name in self.bottleneck:
            x = self.bottleneck[module_name](x)

        for module_name in self.dec_blocks:
            if module_name.startswith("upsampling"):
                x = self.dec_blocks[module_name](x)
                skip_connection = self.skip_layers_ouputs[
                    module_name.replace("upsampling", "skip_encoder_conv")
                ]
                x = torch.cat((x, skip_connection), axis=1)
            else:
                x = self.dec_blocks[module_name](x)
        x = self.final_conv(x)
        x = torch.add(x, inputs)
        return x