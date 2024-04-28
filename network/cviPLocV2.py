import torch
import torch.nn as nn
import torch.nn.functional as F
from tools.utils import IntermediateLayerGetter
from network.backbone.vip import vip_s16


class CViPLocV2(nn.Module):
    """
        ConvNeXt-tiny + 2 ViP
    """

    def __init__(self, feature_extractor, droprate=0.5, pretrained=True, feat_dim=384):
        super(CViPLocV2, self).__init__()
        self.droprate = droprate

        # Use the first four stages in feature_extractor
        return_layers = {'5': 'layer5'}

        # feature_extractor_out_channel = 384

        # backbone
        self.feature_extractor = IntermediateLayerGetter(feature_extractor, return_layers=return_layers)

        # vip_s16
        self.vip_block1 = vip_s16()
        self.vip_block2 = vip_s16()

        self.vip_block1.reset_classifier(num_classes=feat_dim)
        self.vip_block2.reset_classifier(num_classes=feat_dim)

        self.fc_him1 = nn.Linear(feat_dim, 64)
        self.fc_him2 = nn.Linear(feat_dim, 64)

        self.fc_xyz = nn.Linear(64, 3)
        self.fc_wpqr = nn.Linear(64, 3)

        # initialize
        if pretrained:
            init_modules = [self.vip_block1, self.vip_block2, self.fc_xyz, self.fc_wpqr]
        else:
            init_modules = self.modules()

        for m in init_modules:
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        features = self.feature_extractor(x)
        x = features['layer5']
        x = F.gelu(x)
        xyz_block1 = self.vip_block1(x)
        wpqr_block2 = self.vip_block2(x)

        if self.droprate > 0:
            xyz_block1 = F.dropout(xyz_block1, p=self.droprate)
            wpqr_block2 = F.dropout(wpqr_block2, p=self.droprate)

        xyz_block1 = F.gelu(self.fc_him1(xyz_block1))
        wpqr_block2 = F.gelu(self.fc_him2(wpqr_block2))

        xyz = self.fc_xyz(xyz_block1)
        wpqr = self.fc_wpqr(wpqr_block2)
        return torch.cat((xyz, wpqr), 1)





