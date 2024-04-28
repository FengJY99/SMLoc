import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
from network.coordatt import CoordAtt
from torchvision import models
from torch import nn, Tensor

from collections import OrderedDict
from typing import Dict, List
from functools import partial


class IntermediateLayerGetter(nn.ModuleDict):
    """
        Module wrapper that returns intermediate layers from a model

        It has a strong assumption that the modules have been registered
        into the model in the same order as they are used.
        This means that one should **not** reuse the same nn.Module
        twice in the forward if you want this to work.

        Additionally, it is only able to query submodules that are directly
        assigned to the model. So if `model` is passed, `model.feature1` can
        be returned, but not `model.feature1.layer2`.

        Args:
            model (nn.Module): model on which we will extract the features
            return_layers (Dict[name, new_name]): a dict containing the names
                of the modules for which the activations will be returned as
                the key of the dict, and the value of the dict is the name
                of the returned activation (which the user can specify).
    """
    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        # 因为有些backbone的name是序号，是数值型
        return_layers = {str(k): str(v) for k, v in return_layers.items()}

        # 重新构建backbone，将没有使用到的模块全部删掉
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        # 将通道维度调整到最后
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        # 再将通道维度调整回来
        x = x.permute(0, 3, 1, 2)
        return x


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        super(ASPPConv, self).__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.LayerNorm(out_channels, 1e-6),
            nn.GELU()
        )


# class ASPPPooling(nn.Sequential):
#     def __init__(self, in_channels: int, out_channels: int) -> None:
#         super(ASPPPooling, self).__init__(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(in_channels, out_channels, 1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU()
#         )
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         size = x.shape[-2:]
#         for mod in self:
#             x = mod(x)
#         return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels: int, atrous_rates: List[int], droprate: float = 0.5, out_channels: int = 384) -> None:
        super(ASPP, self).__init__()
        self.droprate = droprate
        modules = [
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                          nn.LayerNorm(out_channels, eps=1e-6),
                          nn.GELU())
        ]

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        # 去掉最后一个分支
        # modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.fuse = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, len(self.convs) * out_channels, kernel_size=2, stride=2, bias=False),
            nn.LayerNorm(out_channels, eps=1e-6),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        res = self.fuse(res)

        if self.droprate > 0:
            res = F.dropout(res, p=self.droprate)
        return res


class CooAttLoc(nn.Module):
    def __init__(self, feature_extractor, droprate=0.5, pretrained=True, feat_dim=2048, cooAtt=True):
        super(CooAttLoc, self).__init__()
        self.droprate = droprate
        self.is_cooAtt = cooAtt
        # atrous_rates = [1, 2, 4]
        feature_extractor_out_channel = 384
        # python的高阶函数用法，将LayerNorm2d函数中eps=这个参数固定值为1e-6
        norm_layer = partial(LayerNorm2d, eps=1e-6)

        # Use the first three stages in feature_extractor
        return_layers = {'5': 'out'}
         # downsample 16×
        self.feature_extractor = IntermediateLayerGetter(feature_extractor, return_layers=return_layers)

        # Coordinate Attention
        self.cooAtt = CoordAtt(feature_extractor_out_channel, feature_extractor_out_channel)

        self.feature_cooAtt_avgpool = nn.AdaptiveAvgPool2d(1)

        self.feature_fc = nn.Sequential(
            norm_layer(384), nn.Flatten(1),
            nn.Linear(384, feat_dim)
        )

        self.fc_xyz = nn.Linear(feat_dim, 3)
        self.fc_wpqr = nn.Linear(feat_dim, 3)

        # initialize
        if pretrained:
            init_modules = [self.cooAtt, self.feature_fc, self.fc_xyz, self.fc_wpqr]
        else:
            init_modules = self.modules()

        for m in init_modules:
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        features = self.feature_extractor(x)
        x = features['out']
        # x = F.gelu(x)
        x = self.cooAtt(x)

        x = self.feature_cooAtt_avgpool(x)

        x = F.gelu(self.feature_fc(x))

        if self.droprate > 0:
            x = F.dropout(x, p=self.droprate)

        xyz = self.fc_xyz(x)
        wpqr = self.fc_wpqr(x)
        return torch.cat((xyz, wpqr), 1)
