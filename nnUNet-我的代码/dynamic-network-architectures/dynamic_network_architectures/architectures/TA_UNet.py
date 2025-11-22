
from typing import Union, Type, List, Tuple
import torch
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list, get_matching_pool_op
import numpy as np
from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp

#定义 TemporalAttention 模块
class TemporalAttention(nn.Module):
    """
    Temporal Attention module for fusing information from the current and previous frames.
    Args:
        channels (int): Number of channels in the input feature maps.
        kernel_size (int): Size of the convolution kernel.
    """

    def __init__(self, channels, n_stages: int, kernel_size: Union[int, List[int], Tuple[int, ...]]):
        super(TemporalAttention, self).__init__()
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * n_stages
        # 修改卷积层的输入通道数为 2 * channels
        # 处理 padding，确保对列表中的每个元素都进行 // 2 操作
        if isinstance(kernel_size, (list, tuple)):
            padding = [k // 2 for k in kernel_size]
        else:
            padding = kernel_size // 2
        self.conv = nn.Conv3d(2 * channels, 1, kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()

    def forward(self, skip_current, skip_previous):
        x = torch.cat([skip_current, skip_previous], dim=1)
        att = self.sigmoid(self.conv(x))
        skip_fused = att * skip_current
        return skip_fused

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
        nonlin_first: 如果为True，则顺序为 conv -> nonlin -> norm。否则为 conv -> norm -> nonlin
        """
        super().__init__()
        # 如果 n_conv_per_stage 是整数，则将其扩展为与 n_stages 相同长度的列表
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        # 如果 n_conv_per_stage_decoder 是整数，则将其扩展为与 n_stages - 1 相同长度的列表
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)

        # 检查 n_conv_per_stage 和 n_conv_per_stage_decoder 的长度是否正确
        assert len(n_conv_per_stage) == n_stages, "n_conv_per_stage 必须与分辨率阶段数相同"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder 必须比分辨率阶段数少一个"

        # 初始化两个编码器，分别处理当前帧和前一帧
        self.encoder_current = PlainConvEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                                  n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                                  dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
                                                  nonlin_first=nonlin_first)
        self.encoder_previous = PlainConvEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                                   n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                                   dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
                                                   nonlin_first=nonlin_first)

        # 初始化解码器
        self.decoder = UNetDecoder(self.encoder_current, num_classes, n_conv_per_stage_decoder, deep_supervision,
                                     nonlin_first=nonlin_first)
        print('\r\n*********************************************************************************************************\r\n'
              '使用 TA UNet...\r\n'
              '*********************************************************************************************************\r\n')

    def forward(self, x_current, x_previous):
        # 分别对当前帧和前一帧进行编码
        skips_current = self.encoder_current(x_current)
        skips_previous = self.encoder_previous(x_previous)
        # 解码器处理融合后的跳跃连接并输出结果
        return self.decoder(skips_current, skips_previous)

    def compute_conv_feature_map_size(self, input_size):
        # 计算卷积特征图的大小
        assert len(input_size) == convert_conv_op_to_dim(self.encoder_current.conv_op), "输入尺寸应为图像尺寸，不包括颜色/特征通道或批次通道"
        return self.encoder_current.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(
            input_size)

    @staticmethod
    def initialize(module):
        # 初始化权重
        InitWeights_He(1e-2)(module)


class PlainConvEncoder(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 return_skips: bool = False,
                 nonlin_first: bool = False,
                 pool: str = 'conv'):
        super().__init__()
        # 将单个值扩展为列表
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages

        # 检查参数长度是否正确
        assert len(kernel_sizes) == n_stages, "kernel_sizes 必须与分辨率阶段数相同"
        assert len(n_conv_per_stage) == n_stages, "n_conv_per_stage 必须与分辨率阶段数相同"
        assert len(features_per_stage) == n_stages, "features_per_stage 必须与分辨率阶段数相同"
        assert len(strides) == n_stages, "strides 必须与分辨率阶段数相同"

        stages = []
        for s in range(n_stages):
            stage_modules = []
            if pool == 'max' or pool == 'avg':
                if (isinstance(strides[s], int) and strides[s] != 1) or \
                        isinstance(strides[s], (tuple, list)) and any([i != 1 for i in strides[s]]):
                    stage_modules.append(
                        get_matching_pool_op(conv_op, pool_type=pool)(kernel_size=strides[s], stride=strides[s]))
                conv_stride = 1
            elif pool == 'conv':
                conv_stride = strides[s]
            else:
                raise RuntimeError("不支持的池化类型")

            stage_modules.append(
                StackedConvBlocks(n_conv_per_stage[s], conv_op, input_channels, features_per_stage[s],
                                    kernel_sizes[s], conv_stride,
                                    conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin,
                                    nonlin_kwargs, nonlin_first))
            stages.append(nn.Sequential(*stage_modules))
            input_channels = features_per_stage[s]

        self.stages = nn.Sequential(*stages)
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips

        # 存储一些潜在解码器需要的信息
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes

    def forward(self, x):
        ret = []
        for s in self.stages:
            x = s(x)
            ret.append(x)
        if self.return_skips:
            return ret
        else:
            return ret[-1]

    def compute_conv_feature_map_size(self, input_size):
        output = np.int64(0)
        for s in range(len(self.stages)):
            if isinstance(self.stages[s], nn.Sequential):
                for sq in self.stages[s]:
                    if hasattr(sq, 'compute_conv_feature_map_size'):
                        output += self.stages[s][-1].compute_conv_feature_map_size(input_size)
            else:
                output += self.stages[s].compute_conv_feature_map_size(input_size)
            input_size = [i // j for i, j in zip(input_size, self.strides[s])]
        return output


class UNetDecoder(nn.Module):
    def __init__(self,
                 encoder: Union[PlainConvEncoder],
                 num_classes: int,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision,
                 nonlin_first: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 conv_bias: bool = None):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)

        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)

        assert len(n_conv_per_stage) == n_stages_encoder - 1, "n_conv_per_stage must match n_stages_encoder - 1"

        transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)
        conv_bias = encoder.conv_bias if conv_bias is None else conv_bias
        norm_op = encoder.norm_op if norm_op is None else norm_op
        norm_op_kwargs = encoder.norm_op_kwargs if norm_op_kwargs is None else norm_op_kwargs
        dropout_op = encoder.dropout_op if dropout_op is None else dropout_op
        dropout_op_kwargs = encoder.dropout_op_kwargs if dropout_op_kwargs is None else dropout_op_kwargs
        nonlin = encoder.nonlin if nonlin is None else nonlin
        nonlin_kwargs = encoder.nonlin_kwargs if nonlin_kwargs is None else nonlin_kwargs

        stages = []
        transpconvs = []
        seg_layers = []

        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_transpconv = encoder.strides[-s]

            transpconvs.append(
                transpconv_op(input_features_below, input_features_skip, stride_for_transpconv, stride_for_transpconv,
                              bias=conv_bias))

            stages.append(StackedConvBlocks(n_conv_per_stage[s - 1], encoder.conv_op, 2 * input_features_skip,
                                              input_features_skip,
                                              encoder.kernel_sizes[-(s + 1)], 1, conv_bias, norm_op, norm_op_kwargs,
                                              dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first))

            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.seg_layers = nn.ModuleList(seg_layers)

        # Correctly initialize TemporalAttention modules
        self.temporal_attentions = nn.ModuleList([
            TemporalAttention(
                channels=encoder.output_channels[-(s + 2)],
                n_stages=n_stages_encoder,
                kernel_size=encoder.kernel_sizes[-(s + 2)]
            )
            for s in range(len(self.stages))
        ])

    def forward(self, skips_current, skips_previous):
        lres_input_current = skips_current[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x_current = self.transpconvs[s](lres_input_current)
            skip_current = skips_current[-(s + 2)]
            skip_previous = skips_previous[-(s + 2)]
            skip_fused = self.temporal_attentions[s](skip_current, skip_previous)
            x = torch.cat((x_current, skip_fused), 1)
            x = self.stages[s](x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input_current = x

        seg_outputs = seg_outputs[::-1]
        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r

    def compute_conv_feature_map_size(self, input_size):
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
        input_size = skip_sizes[-1]

        assert len(skip_sizes) == len(self.stages)

        output = np.int64(0)
        for s in range(len(self.stages)):
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s + 1)])
            output += np.prod([self.encoder.output_channels[-(s + 2)], *skip_sizes[-(s + 1)]], dtype=np.int64)
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += np.prod([self.num_classes, *skip_sizes[-(s + 1)]], dtype=np.int64)
        return output


class StackedConvBlocks(nn.Module):
    def __init__(self,
                 num_convs: int,
                 conv_op: Type[_ConvNd],
                 input_channels: int,
                 output_channels: Union[int, List[int], Tuple[int, ...]],
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 initial_stride: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 nonlin_first: bool = False):
        super().__init__()
        if not isinstance(output_channels, (tuple, list)):
            output_channels = [output_channels] * num_convs

        self.convs = nn.Sequential(
            ConvDropoutNormReLU(
                conv_op, input_channels, output_channels[0], kernel_size, initial_stride, conv_bias, norm_op,
                norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first
            ),
            *[
                ConvDropoutNormReLU(
                    conv_op, output_channels[i - 1], output_channels[i], kernel_size, 1, conv_bias, norm_op,
                    norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first
                )
                for i in range(1, num_convs)
            ]
        )
        
        self.act = nonlin(**nonlin_kwargs)
        self.output_channels = output_channels[-1]
        self.initial_stride = maybe_convert_scalar_to_list(conv_op, initial_stride)

    def forward(self, x):
        out = self.convs(x)
        # out = self.act(out)
        return out

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == len(self.initial_stride), "输入尺寸应为图像尺寸，不包括颜色/特征通道或批次通道"
        output = self.convs[0].compute_conv_feature_map_size(input_size)
        size_after_stride = [i // j for i, j in zip(input_size, self.initial_stride)]
        for b in self.convs[1:]:
            output += b.compute_conv_feature_map_size(size_after_stride)
        return output


class ConvDropoutNormReLU(nn.Module):
    def __init__(self,
                 conv_op: Type[_ConvNd],
                 input_channels: int,
                 output_channels: int,
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 stride: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 nonlin_first: bool = False):
        super(ConvDropoutNormReLU, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        stride = maybe_convert_scalar_to_list(conv_op, stride)
        self.stride = stride
        kernel_size = maybe_convert_scalar_to_list(conv_op, kernel_size)

        if norm_op_kwargs is None:
            norm_op_kwargs = {}
        if nonlin_kwargs is None:
            nonlin_kwargs = {}

        ops = []
        self.conv = conv_op(input_channels, output_channels, kernel_size, stride,
                            padding=[(i - 1) // 2 for i in kernel_size], dilation=1, bias=conv_bias)
        ops.append(self.conv)

        if dropout_op is not None:
            self.dropout = dropout_op(**dropout_op_kwargs)
            ops.append(self.dropout)

        if norm_op is not None:
            self.norm = norm_op(output_channels, **norm_op_kwargs)
            ops.append(self.norm)

        if nonlin is not None:
            self.nonlin = nonlin(**nonlin_kwargs)
            ops.append(self.nonlin)

        if nonlin_first and (norm_op is not None and nonlin is not None):
            ops[-1], ops[-2] = ops[-2], ops[-1]

        self.all_modules = nn.Sequential(*ops)

    def forward(self, x):
        return self.all_modules(x)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == len(self.stride), "输入尺寸应为图像尺寸，不包括颜色/特征通道或批次通道"
        output_size = [i // j for i, j in zip(input_size, self.stride)]
        return np.prod([self.output_channels, *output_size], dtype=np.int64)


class ConvDropoutNorm(nn.Module):
    def __init__(self,
                 conv_op: Type[_ConvNd],
                 input_channels: int,
                 output_channels: int,
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 stride: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 nonlin_first: bool = False):
        super(ConvDropoutNorm, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        stride = maybe_convert_scalar_to_list(conv_op, stride)
        self.stride = stride
        kernel_size = maybe_convert_scalar_to_list(conv_op, kernel_size)

        if norm_op_kwargs is None:
            norm_op_kwargs = {}
        if nonlin_kwargs is None:
            nonlin_kwargs = {}

        ops = []
        self.conv = conv_op(input_channels, output_channels, kernel_size, stride,
                            padding=[(i - 1) // 2 for i in kernel_size], dilation=1, bias=conv_bias)
        ops.append(self.conv)

        if dropout_op is not None:
            self.dropout = dropout_op(**dropout_op_kwargs)
            ops.append(self.dropout)

        if norm_op is not None:
            self.norm = norm_op(output_channels, **norm_op_kwargs)
            ops.append(self.norm)

        self.all_modules = nn.Sequential(*ops)

    def forward(self, x):
        return self.all_modules(x)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == len(self.stride), "输入尺寸应为图像尺寸，不包括颜色/特征通道或批次通道"
        output_size = [i // j for i, j in zip(input_size, self.stride)]
        return np.prod([self.output_channels, *output_size], dtype=np.int64)

