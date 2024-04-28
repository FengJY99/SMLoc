import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
from functools import partial
from network.atloc import FourDirectionalLSTM
from network.att import AttentionBlock


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class ConvAtLoc(nn.Module):

    def __init__(self, feature_extractor, droprate=0.5, pretrained=True, feat_dim=768, lstm=False):

        super(ConvAtLoc, self).__init__()
        self.droprate = droprate
        self.lstm = lstm

        # ConvNeXt
        self.feature_extractor = feature_extractor
        norm_layer = partial(LayerNorm2d, eps=1e-6)
        self.feature_extractor.classifier = nn.Sequential(
            norm_layer(768), nn.Flatten(1),
            nn.Linear(768, feat_dim)
        )

        if self.lstm:
            self.lstm4dir = FourDirectionalLSTM(seq_size=32, origin_feat_size=feat_dim, hidden_size=256)
            self.fc_xyz = nn.Linear(feat_dim // 2, 3)
            self.fc_wpqr = nn.Linear(feat_dim // 2, 3)
        else:
            self.att = AttentionBlock(feat_dim)
            self.fc_xyz = nn.Linear(feat_dim, 3)
            self.fc_wpqr = nn.Linear(feat_dim, 3)

        # initialize
        if pretrained:
            init_modules = [self.feature_extractor.classifier, self.fc_xyz, self.fc_wpqr]
        else:
            init_modules = self.modules()

        for m in init_modules:
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = F.gelu(x)

        if self.lstm:
            x = self.lstm4dir(x)
        else:
            x = self.att(x.view(x.size(0), -1))

        if self.droprate > 0:
            x = F.dropout(x, p=self.droprate)

        xyz = self.fc_xyz(x)
        wpqr = self.fc_wpqr(x)
        return torch.cat((xyz, wpqr), 1)
