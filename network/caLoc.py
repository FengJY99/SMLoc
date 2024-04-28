import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
from network.coordatt import CoordAtt

from torch import nn, Tensor

from collections import OrderedDict
from typing import Dict, List


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


class CALoc(nn.Module):
    """
        ConvNeXt-tiny + CooAtt
    """

    def __init__(self, feature_extractor, droprate=0.5, pretrained=True, feat_dim=768):
        super(CALoc, self).__init__()
        self.droprate = droprate

        # Use the first four stages in feature_extractor
        return_layers = {'5': 'layer5'}

        feature_extractor_out_channel = 384


        # backbone
        self.feature_extractor = IntermediateLayerGetter(feature_extractor, return_layers=return_layers)

        # Coordinate Attention
        self.cooAtt = CoordAtt(feature_extractor_out_channel, feature_extractor_out_channel)

        # Down sample
        self.down_sample = nn.Sequential(
            LayerNorm2d(feature_extractor_out_channel, eps=1e-6),
            nn.Conv2d(feature_extractor_out_channel, feat_dim, kernel_size=2, stride=2),
            nn.GELU(),
        )
        # GAP
        self.feature_avgpool = nn.AdaptiveAvgPool2d(1)

        self.feature_fc = nn.Sequential(nn.Flatten(1),
                                        nn.Linear(feat_dim, feat_dim),
                                        nn.GELU(),
                                        )

        self.fc_xyz = nn.Linear(feat_dim, 3)
        self.fc_wpqr = nn.Linear(feat_dim, 3)

        # initialize
        if pretrained:
            init_modules = [self.cooAtt, self.down_sample, self.feature_fc, self.fc_xyz, self.fc_wpqr]
        else:
            init_modules = self.modules()

        for m in init_modules:
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        features = self.feature_extractor(x)
        x = features['layer5']
        x = self.cooAtt(x)
        x = self.down_sample(x)
        x = self.feature_avgpool(x)
        x = self.feature_fc(x)

        if self.droprate > 0:
            x = F.dropout(x, p=self.droprate)
        xyz = self.fc_xyz(x)
        wpqr = self.fc_wpqr(x)
        return torch.cat((xyz, wpqr), 1)