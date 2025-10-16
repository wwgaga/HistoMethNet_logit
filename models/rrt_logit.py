from torch import nn
import torch.nn.functional as F

from models.rrt import RRTEncoder, initialize_weights
from models.datten import DAttention
from models.logit_regression import LogitRegression


class RRTMILLogit(nn.Module):
    """
    RRTMIL variant that replaces the TransformerBackbone classifier
    with a simple multinomial logistic regression head (single Linear layer).

    Interface mirrors models.rrt.RRTMIL for easy drop-in.
    """

    def __init__(
        self,
        input_dim: int = 1536,
        mlp_dim: int = 512,
        act: str = 'relu',
        n_classes: int = 2,
        dropout: float = 0.25,
        pos_pos: int = 0,
        pos: str = 'none',
        peg_k: int = 7,
        attn: str = 'rmsa',
        pool: str = 'attn',
        region_num: int = 8,
        n_layers: int = 2,
        n_heads: int = 8,
        drop_path: float = 0.0,
        da_act: str = 'relu',
        trans_dropout: float = 0.1,
        ffn: bool = False,
        ffn_act: str = 'gelu',
        mlp_ratio: float = 4.0,
        da_gated: bool = False,
        da_bias: bool = False,
        da_dropout: bool = False,
        trans_dim: int = 64,
        epeg: bool = True,
        min_region_num: int = 0,
        cell_property: str = 'cell_type',
        qkv_bias: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        # Projection from input features to encoder dim (mirror original RRTMIL)
        input_dim = 1536  # keep consistent with original RRTMIL
        proj = [nn.Linear(input_dim, 512)]
        if act.lower() == 'relu':
            proj += [nn.ReLU()]
        elif act.lower() == 'gelu':
            proj += [nn.GELU()]
        self.patch_to_emb = nn.Sequential(*proj)
        self.dp = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Encoder (unchanged)
        self.online_encoder = RRTEncoder(
            mlp_dim=mlp_dim,
            pos_pos=pos_pos,
            pos=pos,
            peg_k=peg_k,
            attn=attn,
            region_num=region_num,
            n_layers=n_layers,
            n_heads=n_heads,
            drop_path=drop_path,
            drop_out=trans_dropout,
            ffn=ffn,
            ffn_act=ffn_act,
            mlp_ratio=mlp_ratio,
            trans_dim=trans_dim,
            epeg=epeg,
            min_region_num=min_region_num,
            qkv_bias=qkv_bias,
            **kwargs,
        )

        # Pooling placeholder (not used by this head, kept for compatibility)
        self.pool_fn = (
            DAttention(self.online_encoder.final_dim, da_act, gated=da_gated, bias=da_bias, dropout=da_dropout)
            if pool == 'attn'
            else nn.AdaptiveAvgPool1d(1)
        )

        # Simple multinomial logistic regression classifier
        if cell_property == 'cell_type':
            num_classes = 8
        else:
            num_classes = 5
        self.cell_type_classifiers = LogitRegression(self.online_encoder.final_dim, num_classes)

        # Weight initialization consistent with original code
        self.apply(initialize_weights)

    def forward(self, x, return_attn: bool = False, no_norm: bool = False):
        # x: [num_patches, input_dim]
        x = self.patch_to_emb(x)
        x = self.dp(x)

        # Encoder features per patch
        features = self.online_encoder(x)

        # Linear logits per patch
        patch_level_logits_cell_type = self.cell_type_classifiers(features)

        # Slide-level probability: average per-patch softmax
        cell_type_prob = F.softmax(patch_level_logits_cell_type, dim=1).mean(dim=0)

        return {
            'cell_type_logits': patch_level_logits_cell_type,
            'cell_type_prob': cell_type_prob,
            'features': features,
        }
