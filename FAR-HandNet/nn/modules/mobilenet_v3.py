from typing import Callable, List, Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from functools import partial
from .fla import FocusedLinearAttention

def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


class ConvBNActivation(nn.Sequential):
    def __init__(self,
                 in_planes: int,
                 out_planes: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 activation_layer: Optional[Callable[..., nn.Module]] = None,
                 groups: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        if activation_layer == 'HS':
            activation_layer = nn.Hardswish
        super(ConvBNActivation, self).__init__(nn.Conv2d(in_channels=in_planes,
                                                         out_channels=out_planes,
                                                         kernel_size=kernel_size,
                                                         stride=stride,
                                                         padding=padding,
                                                         groups=groups,
                                                         bias=False),
                                               norm_layer(out_planes),
                                               activation_layer(inplace=True))


class SqueezeExcitation(nn.Module):
    def __init__(self, input_c: int, squeeze_factor: int = 4):
        super(SqueezeExcitation, self).__init__()
        squeeze_c = _make_divisible(input_c // squeeze_factor, 8)
        self.fc1 = nn.Conv2d(input_c, squeeze_c, 1)
        self.fc2 = nn.Conv2d(squeeze_c, input_c, 1)

    def forward(self, x: Tensor) -> Tensor:
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        scale = self.fc1(scale)
        scale = F.relu(scale, inplace=True)
        scale = self.fc2(scale)
        scale = F.hardsigmoid(scale, inplace=True)
        return scale * x



class InvertedResidual(nn.Module):
    def __init__(self,input_c: int, out_c: int, expanded_c: int, kernel: int,
                 stride: int, activation: str, use_attention: str,
                 norm_layer: Optional[Callable[..., nn.Module]] = None):
        super(InvertedResidual, self).__init__()
        self.input_c = input_c
        self.kernel = kernel
        self.expanded_c = expanded_c
        self.out_c = out_c
        self.use_se = use_attention == "SE"
        self.use_hs = activation == "HS"  # whether using h-swish activation
        self.use_fla = use_attention == "FLA"
        self.stride = stride

        if self.stride not in [1, 2]:
            raise ValueError("illegal stride value.")
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.use_res_connect = (self.stride == 1 and self.input_c == self.out_c)

        layers: List[nn.Module] = []
        activation_layer = nn.Hardswish if self.use_hs else nn.ReLU

        # expand
        if self.expanded_c != self.input_c:
            layers.append(ConvBNActivation(self.input_c,
                                           self.expanded_c,
                                           kernel_size=1,
                                           norm_layer=norm_layer,
                                           activation_layer=activation_layer))

        # depthwise
        layers.append(ConvBNActivation(self.expanded_c,
                                       self.expanded_c,
                                       kernel_size=self.kernel,
                                       stride=self.stride,
                                       groups=self.expanded_c,
                                       norm_layer=norm_layer,
                                       activation_layer=activation_layer))

        if self.use_se:
            layers.append(SqueezeExcitation(self.expanded_c))

        if self.use_fla:
            layers.append(FocusedLinearAttention(self.expanded_c))

        # project
        layers.append(ConvBNActivation(self.expanded_c,
                                       self.out_c,
                                       kernel_size=1,
                                       norm_layer=norm_layer,
                                       activation_layer=nn.Identity))

        self.block = nn.Sequential(*layers)
        self.out_channels = self.out_c
        self.is_strided = self.stride > 1

    def forward(self, x: Tensor) -> Tensor:
        result = self.block(x)
        if self.use_res_connect:
            result += x

        return result

