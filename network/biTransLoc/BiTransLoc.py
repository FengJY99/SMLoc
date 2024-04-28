"""

coding:utf-8
@Author: HaoNan Sun
@Create: 2024/01/01 23:29
    以ViT作为backbone构建PoseNet式的位姿估计模型-BiTransLoc
"""
import torch
import torch.nn.functional as F
from torch import nn
from network.biTransLoc.vit import VisionTransformer

class BiTransLoc(nn.Module):

    def __init__(self, config):
        super(BiTransLoc, self).__init__()


        self.vit = VisionTransformer(config)
        hidden_dim = self.vit.hidden_dim

        self.regressor_head_t = PoseRegressor(hidden_dim, 3)
        self.regressor_head_rot = PoseRegressor(hidden_dim, 3)


    def forward(self, x):
        x = self.vit(x)
        x_t = self.regressor_head_t(x)
        x_rot = self.regressor_head_rot(x)

        return torch.cat((x_t, x_rot), 1)

class PoseRegressor(nn.Module):
    """ A simple MLP to regress a pose component"""

    def __init__(self, decoder_dim, output_dim):
        """
        decoder_dim: (int) the input dimension
        output_dim: (int) the output dimension
        use_prior: (bool) whether to use prior information
        """
        super().__init__()
        ch = 1024
        self.fc_h = nn.Linear(decoder_dim, ch)
        self.fc_o = nn.Linear(ch, output_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """
        Forward pass
        """
        x = F.gelu(self.fc_h(x))
        return self.fc_o(x)
