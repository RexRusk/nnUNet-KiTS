from typing import Union, Type, List, Tuple
import torch
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from torch.nn import init
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
from dynamic_network_architectures.initialization.weight_init import InitWeights_He

class ChanAttention(nn.Module):
    def __init__(self, C):
        super(ChanAttention, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear1 = nn.Linear(C, C // 4, bias=False)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(C // 4, C, bias=False)
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x)
        y = self.linear1(y.view(b, c))
        y = self.sigm(self.linear2(self.relu(y))).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.sigm = nn.Sigmoid()
        self.conv = ConvBlock(2, 1, 3, 1, 1)

    def forward(self, x):
        x1 = torch.mean(x, dim=1, keepdim=True)
        x2, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat((x1, x2), dim=1)
        out = self.conv(out)
        return self.sigm(out) * x


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k_s, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, k_s, stride=stride, padding=padding)
        self.batchnorm = nn.BatchNorm2d(out_c)
        self.relu = nn.LeakyReLU()
        self.init_weight()

    def forward(self, x):
        x = self.relu(self.batchnorm(self.conv(x)))
        return x

    def init_weight(self):
        # 使用 Kaiming 正态分布初始化卷积层的权重
        init.kaiming_normal_(self.conv.weight, a=0)
        # 如果卷积层有偏置，则初始化为 0
        if self.conv.bias is not None:
            init.constant_(self.conv.bias, 0)

        init.constant_(self.batchnorm.weight, 0)
        init.constant_(self.batchnorm.bias, 0)




class PlainConvUNet(nn.Module):
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
                 nonlin_first: bool = False
                 ):
        """
        nonlin_first: if True you get conv -> nonlin -> norm. Else it's conv -> norm -> nonlin
        """
        super().__init__()

        # here
        channel_attentions = []
        spatial_attentions = []
        for i in range(len(features_per_stage)):
            chan_att = ChanAttention(features_per_stage[i])
            spa_att = SpatialAttention()
            channel_attentions.append(chan_att)
            spatial_attentions.append(spa_att)

        self.chan_atten = nn.Sequential(*channel_attentions)
        self.spa_atten = nn.Sequential(*spatial_attentions)


        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_conv_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have " \
                                                  f"resolution stages. here: {n_stages}. " \
                                                  f"n_conv_per_stage: {n_conv_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        self.encoder = PlainConvEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                        n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                        dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
                                        nonlin_first=nonlin_first)
        self.decoder = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision,
                                   nonlin_first=nonlin_first)

    def forward(self, x):

        '''skips = self.encoder(x)
        return self.decoder(skips)'''

        skips = self.encoder(x)
        # 加入注意力机制
        for i in range(len(skips)):
            skips[i] = self.chan_atten[i](skips[i])
            skips[i] = self.spa_atten[i](skips[i])
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                                            "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                            "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)



if __name__ == '__main__':
    data = torch.rand((1, 4, 128, 128, 128))

    model = PlainConvUNet(4, 6, (32, 64, 128, 256, 320, 320), nn.Conv3d, 3, ((1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2),(1,2,2), (1,2,2)), (2, 2, 2, 2, 2, 2), 4,
                                (2, 2, 2, 2, 2), False, nn.BatchNorm3d, None, None, None, nn.ReLU, deep_supervision=True)

    if False:
        import hiddenlayer as hl

        g = hl.build_graph(model, data,
                           transforms=None)
        g.save("network_architecture.pdf")
        del g

    print(model.compute_conv_feature_map_size(data.shape[2:]))
