import torch
from typing import Union, Type, List, Tuple
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd


class SegNet(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = nn.ReLU,
                 nonlin_kwargs: dict = {'inplace': True},
                 deep_supervision: bool = False,
                 nonlin_first: bool = False
                 ):
        super().__init__()

        # Ensure lists for stages
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages

        # Encoder (downsampling)
        self.encoder_stages = nn.ModuleList()
        current_channels = input_channels
        for i in range(n_stages):
            layer = []
            for j in range(n_conv_per_stage[i]):
                layer.append(
                    conv_op(current_channels, features_per_stage[i], kernel_size=kernel_sizes[i], stride=strides[i],
                            bias=conv_bias))
                if norm_op:
                    layer.append(norm_op(features_per_stage[i], **norm_op_kwargs))
                if nonlin:
                    layer.append(nonlin(**nonlin_kwargs))
                current_channels = features_per_stage[i]
            if i < n_stages - 1:  # Add pooling only before the last stage
                layer.append(nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True))
            self.encoder_stages.append(nn.Sequential(*layer))

        # Bottleneck
        self.bottleneck = nn.Sequential(
            conv_op(current_channels, features_per_stage[-1], kernel_size=kernel_sizes[-1], stride=1, bias=conv_bias),
            norm_op(features_per_stage[-1], **norm_op_kwargs),
            nonlin(**nonlin_kwargs),
            *[conv_op(features_per_stage[-1], features_per_stage[-1], kernel_size=kernel_sizes[-1], stride=1,
                      bias=conv_bias),
              norm_op(features_per_stage[-1], **norm_op_kwargs),
              nonlin(**nonlin_kwargs)] * 3  # Repeat for additional convolutions in the bottleneck
        )

        # Decoder (upsampling)
        self.decoder_stages = nn.ModuleList()
        for i in reversed(range(n_stages)):
            layer = []
            for _ in range(n_conv_per_stage_decoder[i]):
                layer.append(conv_op(current_channels + features_per_stage[i], features_per_stage[i],
                                     kernel_size=kernel_sizes[i], stride=1, bias=conv_bias))
                if norm_op:
                    layer.append(norm_op(features_per_stage[i], **norm_op_kwargs))
                if nonlin:
                    layer.append(nonlin(**nonlin_kwargs))
            layer.append(nn.MaxUnpool2d(kernel_size=2, stride=2))
            self.decoder_stages.append(nn.Sequential(*layer))
            current_channels += features_per_stage[i]  # For concatenation in the decoder

        # Final classification layer
        self.final_conv = conv_op(current_channels, num_classes, kernel_size=1, bias=True)

    def forward(self, x):
        skip_connections = []
        for stage in self.encoder_stages:
            x, indices = stage(x) if hasattr(stage[-1], 'return_indices') else (stage(x), None)
            skip_connections.append((x, indices))

        x = self.bottleneck(x)

        for stage, (skip_x, indices) in zip(self.decoder_stages, reversed(skip_connections)):
            x = torch.cat([x, skip_x], dim=1)
            x = stage[0:-1](x)  # Apply convolutions and activations
            x = stage[-1](x, indices)  # Unpool

        # Apply final classification layer
        x = self.final_conv(x)
        return x