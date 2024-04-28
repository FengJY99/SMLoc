"""

coding:utf-8
@Author: HaoNan Sun
@Create: 2024/01/02 13:30
"""
import math
from collections import OrderedDict
import torch
import torch.nn.functional as F
from torch import nn, Tensor


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class EncoderBlock(nn.Module):
    def __init__(self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        activation: str,
        norm_layer: None,
    ):
        super().__init__()
        self.num_heads = num_heads

        self.activation = _get_activation_fn(activation)

        # Attention block
        self.ln_1 = norm_layer
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer
        self.linear1 = nn.Linear(hidden_dim, mlp_dim)
        self.linear2 = nn.Linear(mlp_dim, hidden_dim)

    def forward(self, input :torch.Tensor):
        x = self.ln_1(input)
        x, _ = self.self_attention(x, x, x, need_weights=False)

        x = self.dropout(x)
        x = x + input


        y = self.ln_2(x)
        # MLP block
        y = self.activation(self.linear1(y))
        y = self.dropout(y)
        y = self.linear2(y)
        y = self.dropout(y)
        return x + y


class Encoder(nn.Module):

    def __init__(self,seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        activation : str,
        norm_layer: None,
        ):
        super().__init__()

        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length,
                                                      hidden_dim).normal_(std=0.02))  # from BERT
        self.dropout = nn.Dropout(dropout);
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                activation,
                norm_layer,
            )

        self.layers = nn.Sequential(layers);
        self.ln = norm_layer;



    def forward(self, input: torch.Tensor):
        input = input + self.pos_embedding
        return self.ln(self.layers(self.dropout(input)))


class VisionTransformer(nn.Module):
    default_config = {
        "image_size":256,
        "patch_size": 16,
        "num_layers":12,
        "num_heads" : 12,
        "hidden_dim": 768,
        "mlp_dim" : 3072,
        "attention_dropout": 0.0,
        "dropout": 0.1,
        "activation": "gelu",
        "normalize_before": True
    }

    def __init__(self,
                 image_size : 256,
                 patch_size : 16,
                 hidden_dim : 768,
                 mlp_dim : 3072,
                 dropout : 0.1,
                 activation : "gelu",
                 num_layers : 12,
                 num_heads : 12,
                 attention_dropout : 0.0,
                 normalize_before : True,
                 ):
        super().__init__()

        norm_layer = nn.LayerNorm(hidden_dim) if normalize_before else None

        self.conv_proj = nn.Conv2d(
            in_channels=3, out_channels=hidden_dim,
            kernel_size=patch_size, stride=patch_size
        );


        seq_length = (image_size // patch_size) ** 2

        # Add a class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        seq_length += 1

        self.encoder = Encoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            activation,
            norm_layer,
        )

        self.seq_length = seq_length
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim


        if isinstance(self.conv_proj, nn.Conv2d):
            # Init the patchify stem
            fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
            nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
            if self.conv_proj.bias is not None:
                nn.init.zeros_(self.conv_proj.bias)


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size

        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x: torch.Tensor):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        # batch size
        n = x.shape[0]
        # 扩展token在批量上的维度
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.encoder(x)
        # Classifier "token" as used by standard language architectures
        x = x[:, 0]
        return x
