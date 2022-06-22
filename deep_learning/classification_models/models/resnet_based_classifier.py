# This code is an adapted version of Pytorch's ResNet Implementation
import torch
import numpy as np

from collections import OrderedDict
from typing import Type, Callable, Union, List, Optional
from torch import Tensor, nn


def conv3x3(
    in_planes: int, out_planes: int, stride: int = 1,
    groups: int = 1, dilation: int = 1
):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride,
        padding=dilation, groups=groups, bias=False, dilation=dilation
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1):
    """1x1 convolution"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self, inplanes: int, planes: int, stride: int = 1,
        downsample: Optional[nn.Module] = None, groups: int = 1,
        base_width: int = 64, dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        act_fn: Optional[Callable[..., nn.Module]] = None,
        use_middle_act: bool = True
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        if act_fn is None:
            act_fn = nn.ReLU

        self.use_middle_activation = use_middle_act

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.act_fn = act_fn()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        if self.use_middle_activation:
            out = self.act_fn(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act_fn(out)
        return out


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int, planes: int, stride: int = 1,
        downsample: Optional[nn.Module] = None, groups: int = 1,
        base_width: int = 64, dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        act_fn: Optional[Callable[..., nn.Module]] = None
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_fn is None:
            act_fn = nn.ReLU
        width = int(planes * (base_width / 64.0)) * groups

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.act_fn = act_fn()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act_fn(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act_fn(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act_fn(out)

        return out


class ResNetBased(nn.Module):
    def __init__(
        self,
        block: str = 'basic',
        # layers: List[int],
        num_classes: int = 1,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        inplanes: int = 64,
        act_fn: Optional[Callable[..., nn.Module]] = None,
        downsample_blocks: int = 3,
        fc_dims: np.ndarray = None,
        dropout: float = 0,
        use_middle_act: bool = True,
        block_act: Optional[Callable[..., nn.Module]] = None
    ):
        super().__init__()

        if block == 'basic':
            block = BasicBlock
        else:
            block = Bottleneck

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.use_middle_act = use_middle_act

        if act_fn is None:
            act_fn = nn.ReLU
        if block_act is None:
            block_act = nn.ReLU
        self.block_act = block_act

        self.inplanes = inplanes
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.act_fn = act_fn

        self.inner_layers = []
        out_planes = self.inplanes
        for i in range(downsample_blocks):
            if i == 0:
                self.inner_layers.append(
                    (f'layer{i+1}', self._make_layer(block, out_planes * 2, 2))
                )
            else:
                self.inner_layers.append(
                    (f'layer{i+1}',
                     self._make_layer(
                        block, out_planes * 2, 2, stride=2,
                        dilate=replace_stride_with_dilation[0])))
            out_planes = out_planes * 2

        self.inner_layers = nn.Sequential(OrderedDict(self.inner_layers))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.in_features = out_planes
        if fc_dims is not None:
            layers_list = []
            for i in range(len(fc_dims)):
                in_neurons = self.in_features if i == 0 else fc_dims[i-1]
                layers_list = layers_list + [
                    (f'do{i+1}', nn.Dropout(dropout)),
                    (f'fc{i+1}', nn.Linear(in_neurons, fc_dims[i])),
                    (f'act{i+1}', self.act_fn())]
            layers_list.append((f'fc{i+2}', nn.Linear(fc_dims[i], 1)))
            self.classifier = nn.Sequential(OrderedDict(layers_list))
        else:
            self.classifier = nn.Sequential(OrderedDict([
                ('do', nn.Dropout(dropout)),
                ('fc', nn.Linear(self.in_features, num_classes))
            ]))

        if isinstance(act_fn, nn.LeakyReLU):
            nl = 'leaky_relu'
        else:
            nl = 'relu'

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity=nl)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                nn.init.constant_(m.bias.data, 0)

    def _make_layer(
        self, block: Type[Union[BasicBlock, Bottleneck]], planes: int,
        blocks: int, stride: int = 1, dilate: bool = False,
    ):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        # Here we decide whether to use dilation convs or strides convs
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups,
                self.base_width, previous_dilation, norm_layer,
                act_fn=self.block_act, use_middle_act=self.use_middle_act
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    act_fn=self.block_act,
                    use_middle_act=self.use_middle_act
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act_fn()(x)
        x = self.inner_layers(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
