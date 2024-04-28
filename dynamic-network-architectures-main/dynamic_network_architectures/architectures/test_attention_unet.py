from typing import Union, Type, List, Tuple
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
import torch
import torch.nn as nn

from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.factories import Norm


class PlainConvAttentionUNet(nn.Module):
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
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 nonlin_first: bool = False,
                 attention: bool = True,
                 up_kernel_size: int = 3,
                 strides_attention: int = 2,
                 dropout_attention: float = 0.0):
        super().__init__()

        self.encoder = PlainConvEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                        n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                        dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
                                        nonlin_first=nonlin_first)

        self.decoder = AttentionDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision,
                                        attention=attention, up_kernel_size=up_kernel_size,
                                        strides=strides_attention, dropout=dropout_attention,
                                        nonlin_first=nonlin_first)

        self.initialize()  # Initialize weights using He initialization

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)


class AttentionDecoder(nn.Module):
    def __init__(self, encoder, num_classes, n_conv_per_stage_decoder, deep_supervision,
                 attention=True, up_kernel_size=3, strides=2, dropout=0.0, nonlin_first=False):
        super().__init__()

        self.decoder_stages = nn.ModuleList()
        self.attention_layers = nn.ModuleList()

        # Reverse the encoder's feature list for decoder
        reversed_features = encoder.features_per_stage[::-1]

        # Build decoder stages
        for idx, (in_channels, out_channels) in enumerate(zip(reversed_features[:-1], reversed_features[1:])):
            self.decoder_stages.append(
                ConvBlock(encoder.spatial_dims, in_channels, out_channels, kernel_size=up_kernel_size,
                          strides=1, dropout=dropout))  # Adjust for upconvolution

            if attention:
                self.attention_layers.append(AttentionLayer(encoder.spatial_dims, out_channels,
                                                            out_channels, submodule=self.decoder_stages[-1],
                                                            up_kernel_size=up_kernel_size, strides=strides,
                                                            dropout=dropout))

        # Final layer
        self.final_conv = nn.Conv2d(reversed_features[-1], num_classes, kernel_size=1)

        # Deep supervision setup, if enabled
        self.deep_supervision = deep_supervision

    def forward(self, skips):
        x = skips[-1]  # Start from the last encoder skip
        for i, (decoder_block, attention_layer) in enumerate(
                zip(reversed(self.decoder_stages), reversed(self.attention_layers))):
            x = attention_layer(x) + decoder_block(x)  # Combine attention and upconvolution
            if self.deep_supervision and i < len(self.decoder_stages) - 1:  # If not the final stage
                # Perform additional operations for deep supervision if needed
                pass

        x = self.final_conv(x)
        return x

class ConvBlock(nn.Module):

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        strides: int = 1,
        dropout=0.0,
    ):
        super().__init__()
        layers = [
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=None,
                adn_ordering="NDA",
                act="relu",
                norm=Norm.BATCH,
                dropout=dropout,
            ),
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                strides=1,
                padding=None,
                adn_ordering="NDA",
                act="relu",
                norm=Norm.BATCH,
                dropout=dropout,
            ),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_c: torch.Tensor = self.conv(x)
        return x_c


class AttentionLayer(nn.Module):

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        submodule: nn.Module,
        up_kernel_size=3,
        strides=2,
        dropout=0.0,
    ):
        super().__init__()
        self.attention = AttentionBlock(
            spatial_dims=spatial_dims, f_g=in_channels, f_l=in_channels, f_int=in_channels // 2
        )
        self.upconv = UpConv(
            spatial_dims=spatial_dims,
            in_channels=out_channels,
            out_channels=in_channels,
            strides=strides,
            kernel_size=up_kernel_size,
        )
        self.merge = Convolution(
            spatial_dims=spatial_dims, in_channels=2 * in_channels, out_channels=in_channels, dropout=dropout
        )
        self.submodule = submodule

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fromlower = self.upconv(self.submodule(x))
        att = self.attention(g=fromlower, x=x)
        att_m: torch.Tensor = self.merge(torch.cat((att, fromlower), dim=1))
        return att_m


class AttentionBlock(nn.Module):

    def __init__(self, spatial_dims: int, f_int: int, f_g: int, f_l: int, dropout=0.0):
        super().__init__()
        self.W_g = nn.Sequential(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=f_g,
                out_channels=f_int,
                kernel_size=1,
                strides=1,
                padding=0,
                dropout=dropout,
                conv_only=True,
            ),
            Norm[Norm.BATCH, spatial_dims](f_int),
        )

        self.W_x = nn.Sequential(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=f_l,
                out_channels=f_int,
                kernel_size=1,
                strides=1,
                padding=0,
                dropout=dropout,
                conv_only=True,
            ),
            Norm[Norm.BATCH, spatial_dims](f_int),
        )

        self.psi = nn.Sequential(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=f_int,
                out_channels=1,
                kernel_size=1,
                strides=1,
                padding=0,
                dropout=dropout,
                conv_only=True,
            ),
            Norm[Norm.BATCH, spatial_dims](1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU()

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi: torch.Tensor = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class UpConv(nn.Module):

    def __init__(self, spatial_dims: int, in_channels: int, out_channels: int, kernel_size=3, strides=2, dropout=0.0):
        super().__init__()
        self.up = Convolution(
            spatial_dims,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=kernel_size,
            act="relu",
            adn_ordering="NDA",
            norm=Norm.BATCH,
            dropout=dropout,
            is_transposed=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_u: torch.Tensor = self.up(x)
        return x_u


if __name__ == '__main__':

    # add parameters here
    model = PlainConvAttentionUNet(4, 6, (32, 64, 125, 256, 320, 320), nn.Conv3d, 3, (1, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2), 4,
                          (2, 2, 2, 2, 2), False, nn.BatchNorm3d, None, None, None, nn.ReLU)

    if False:
        import hiddenlayer as hl

        g = hl.build_graph(model, data,
                           transforms=None)
        g.save("network_architecture.pdf")
        del g

    print(model.compute_conv_feature_map_size(data.shape[2:]))
