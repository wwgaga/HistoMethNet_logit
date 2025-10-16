from torch import nn
from models.emb_position import *
from models.rmsa import *
from models.nystrom_attention import NystromAttention
from models.datten import DAttention
from timm.models.layers import DropPath
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional




class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device, non_blocking=False):
        cast_tensor = self.tensors.to(device, non_blocking=non_blocking)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device, non_blocking=non_blocking)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def record_stream(self, *args, **kwargs):
        self.tensors.record_stream(*args, **kwargs)
        if self.mask is not None:
            self.mask.record_stream(*args, **kwargs)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


class TransformerBackbone(nn.Module):
    def __init__(
        self,
        num_classes,
        patch_dim,
        dim,
        depth,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        self.patch_dim = patch_dim  # Input dimension from encoder
        self.z_dim = dim  # Hidden dimension
        self.depth = depth  # Store depth for use in forward pass
        
        # Initial projection
        self.patch_to_embedding = nn.Linear(patch_dim, self.z_dim)
        self.dropout = nn.Dropout(emb_dropout)
        
        # Create ModuleList for better organization
        self.mlp_layers = nn.ModuleList()
        self.transformer_layers = nn.ModuleList()
        self.residual_layers = nn.ModuleList()
        
        # Transformer and MLP layers for each depth
        for i in range(depth):
            # MLP block
            self.mlp_layers.append(nn.Sequential(
                nn.Linear(self.z_dim, self.z_dim * 2),
                nn.Tanh(),
                nn.LayerNorm(self.z_dim * 2),
                nn.Dropout(dropout),
                nn.Linear(self.z_dim * 2, self.z_dim),
                nn.Tanh(),
                nn.LayerNorm(self.z_dim),
                nn.Dropout(dropout),
            ))
            
            # Transformer block
            self.transformer_layers.append(nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.z_dim,
                    nhead=8,  # Make sure z_dim is divisible by nhead
                    dim_feedforward=self.z_dim * 4,
                    dropout=dropout
                ),
                num_layers=1
            ))
            
            # Residual connection
            self.residual_layers.append(nn.Linear(patch_dim, self.z_dim))

        # Final patch-level classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.z_dim),
            nn.Linear(self.z_dim, num_classes)
        )

    def forward(self, x):
        # x shape: [num_patches, patch_dim]
        batch_size = x.shape[0]
        
        # Store original input for residual connections
        identity = x
        
        # Initial embedding
        x = self.patch_to_embedding(x)  # [num_patches, z_dim]
        x = self.dropout(x)
        
        # Process through transformer blocks
        for i in range(self.depth):
            # MLP
            x = self.mlp_layers[i](x) + x  # Add residual
            
            # Transformer - expects [seq_len, batch, dim]
            x_trans = x.unsqueeze(1)  # Add batch dimension
            x_trans = self.transformer_layers[i](x_trans)
            x = x_trans.squeeze(1)  # Remove batch dimension
            
            # Residual from original input
            x = x + self.residual_layers[i](identity)
        
        # Get patch-level logits
        logits = self.classifier(x)  # [num_patches, num_classes]
        
        return logits, x  # Return both logits and final features


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            # ref from huggingface
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.Linear):
            # ref from clam
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512,head=8,drop_out=0.1,drop_path=0.,ffn=False,ffn_act='gelu',mlp_ratio=4.,trans_dim=64,attn='rmsa',n_region=8,epeg=False,region_size=0,min_region_num=0,min_region_ratio=0,qkv_bias=True,crmsa_k=3,epeg_k=15,**kwargs):
        super().__init__()

        self.norm = norm_layer(dim)
        self.norm2 = norm_layer(dim) if ffn else nn.Identity()
        if attn == 'ntrans':
            self.attn = NystromAttention(
                dim = dim,
                dim_head = trans_dim,  # dim // 8
                heads = head,
                num_landmarks = 256,    # number of landmarks dim // 2
                pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
                residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
                dropout=drop_out
            )
        elif attn == 'rmsa':
            self.attn = RegionAttntion(
                dim=dim,
                num_heads=head,
                drop=drop_out,
                region_num=n_region,
                head_dim=dim // head,
                epeg=epeg,
                region_size=region_size,
                min_region_num=min_region_num,
                min_region_ratio=min_region_ratio,
                qkv_bias=qkv_bias,
                epeg_k=epeg_k,
                **kwargs
            )
        elif attn == 'crmsa':
            self.attn = CrossRegionAttntion(
                dim=dim,
                num_heads=head,
                drop=drop_out,
                region_num=n_region,
                head_dim=dim // head,
                epeg=epeg,
                region_size=region_size,
                min_region_num=min_region_num,
                min_region_ratio=min_region_ratio,
                qkv_bias=qkv_bias,
                crmsa_k=crmsa_k,
                **kwargs
            )
        else:
            raise NotImplementedError
        # elif attn == 'rrt1d':
        #     self.attn = RegionAttntion1D(
        #         dim=dim,
        #         num_heads=head,
        #         drop=drop_out,
        #         region_num=n_region,
        #         head_dim=trans_dim,
        #         conv=epeg,
        #         **kwargs
        #     )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.ffn = ffn
        act_layer = nn.GELU if ffn_act == 'gelu' else nn.ReLU
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,act_layer=act_layer,drop=drop_out) if ffn else nn.Identity()

    def forward(self,x,need_attn=False):

        x,attn = self.forward_trans(x,need_attn=need_attn)
        
        if need_attn:
            return x,attn
        else:
            return x

    def forward_trans(self, x, need_attn=False):
        attn = None
        
        if need_attn:
            z,attn = self.attn(self.norm(x),return_attn=need_attn)
        else:
            z = self.attn(self.norm(x))

        x = x+self.drop_path(z)

        # FFN
        if self.ffn:
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x,attn

class RRTEncoder(nn.Module):
    def __init__(self,mlp_dim=512,pos_pos=0,pos='none',peg_k=7,attn='rmsa',region_num=8,drop_out=0.1,n_layers=2,n_heads=8,drop_path=0.,ffn=False,ffn_act='gelu',mlp_ratio=4.,trans_dim=64,epeg=True,epeg_k=15,region_size=0,min_region_num=0,min_region_ratio=0,qkv_bias=True,peg_bias=True,peg_1d=False,cr_msa=True,crmsa_k=3,all_shortcut=False,crmsa_mlp=False,crmsa_heads=8,need_init=False,**kwargs):
        super(RRTEncoder, self).__init__()
        
        self.final_dim = mlp_dim

        self.norm = nn.LayerNorm(self.final_dim)
        self.all_shortcut = all_shortcut

        self.layers = []
        for i in range(n_layers-1):
            self.layers += [TransLayer(dim=mlp_dim,head=n_heads,drop_out=drop_out,drop_path=drop_path,ffn=ffn,ffn_act=ffn_act,mlp_ratio=mlp_ratio,trans_dim=trans_dim,attn=attn,n_region=region_num,epeg=epeg,region_size=region_size,min_region_num=min_region_num,min_region_ratio=min_region_ratio,qkv_bias=qkv_bias,epeg_k=epeg_k,**kwargs)]
        self.layers = nn.Sequential(*self.layers)
    
        # CR-MSA
        self.cr_msa = TransLayer(dim=mlp_dim,head=crmsa_heads,drop_out=drop_out,drop_path=drop_path,ffn=ffn,ffn_act=ffn_act,mlp_ratio=mlp_ratio,trans_dim=trans_dim,attn='crmsa',qkv_bias=qkv_bias,crmsa_k=crmsa_k,crmsa_mlp=crmsa_mlp,**kwargs) if cr_msa else nn.Identity()

        # only for ablation
        if pos == 'ppeg':
            self.pos_embedding = PPEG(dim=mlp_dim,k=peg_k,bias=peg_bias,conv_1d=peg_1d)
        elif pos == 'sincos':
            self.pos_embedding = SINCOS(embed_dim=mlp_dim)
        elif pos == 'peg':
            self.pos_embedding = PEG(mlp_dim,k=peg_k,bias=peg_bias,conv_1d=peg_1d)
        else:
            self.pos_embedding = nn.Identity()

        self.pos_pos = pos_pos

        if need_init:
            self.apply(initialize_weights)

    def forward(self, x):
        shape_len = 3
        # for N,C
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            shape_len = 2
        # for B,C,H,W
        if len(x.shape) == 4:
            x = x.reshape(x.size(0),x.size(1),-1)
            x = x.transpose(1,2)
            shape_len = 4

        batch, num_patches, C = x.shape 
        x_shortcut = x

        # PEG/PPEG
        if self.pos_pos == -1:
            x = self.pos_embedding(x)
        
        # R-MSA within region
        for i,layer in enumerate(self.layers.children()):
            if i == 1 and self.pos_pos == 0:
                x = self.pos_embedding(x)
            x = layer(x)

        x = self.cr_msa(x)

        if self.all_shortcut:
            x = x+x_shortcut

        x = self.norm(x)

        if shape_len == 2:
            x = x.squeeze(0)
        elif shape_len == 4:
            x = x.transpose(1,2)
            x = x.reshape(batch,C,int(num_patches**0.5),int(num_patches**0.5))
        return x
    
class RRTMIL(nn.Module):
    def __init__(self, input_dim=1536,mlp_dim=512,act='relu',n_classes=2,dropout=0.25,pos_pos=0,pos='none',peg_k=7,attn='rmsa',pool='attn',region_num=8,n_layers=2,n_heads=8,drop_path=0.,da_act='relu',trans_dropout=0.1,ffn=False,ffn_act='gelu',mlp_ratio=4.,da_gated=False,da_bias=False,da_dropout=False,trans_dim=64,epeg=True,min_region_num=0, cell_property='cell_type',qkv_bias=True,**kwargs):
        super(RRTMIL, self).__init__()
        input_dim = 1536
        self.patch_to_emb = [nn.Linear(input_dim, 512)]

        if act.lower() == 'relu':
            self.patch_to_emb += [nn.ReLU()]
        elif act.lower() == 'gelu':
            self.patch_to_emb += [nn.GELU()]

        self.dp = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        self.patch_to_emb = nn.Sequential(*self.patch_to_emb)

        self.online_encoder = RRTEncoder(mlp_dim=mlp_dim,pos_pos=pos_pos,pos=pos,peg_k=peg_k,attn=attn,region_num=region_num,n_layers=n_layers,n_heads=n_heads,drop_path=drop_path,drop_out=trans_dropout,ffn=ffn,ffn_act=ffn_act,mlp_ratio=mlp_ratio,trans_dim=trans_dim,epeg=epeg,min_region_num=min_region_num,qkv_bias=qkv_bias,**kwargs)

        self.pool_fn = DAttention(self.online_encoder.final_dim,da_act,gated=da_gated,bias=da_bias,dropout=da_dropout) if pool == 'attn' else nn.AdaptiveAvgPool1d(1)
        
        # self.predictor = nn.Linear(self.online_encoder.final_dim, n_classes)
        num_trans_mlp_layers = 1
        if cell_property == 'cell_type':
            
            # self.cell_type_classifiers = nn.Linear(self.online_encoder.final_dim, 8)
            self.cell_type_classifiers = TransformerBackbone(
            8,
            self.online_encoder.final_dim,
            64,
            num_trans_mlp_layers,
            dropout=0.0,
            emb_dropout=0.0,
        )
        else:
            self.cell_type_classifiers = TransformerBackbone(
            5,
            self.online_encoder.final_dim,
            64,
            num_trans_mlp_layers,
            dropout=0.0,
            emb_dropout=0.0,
        )
            # self.cell_type_classifiers = nn.Linear(self.online_encoder.final_dim, 5)

        # self.feature_extractor = nn.Sequential(nn.Linear(self.online_encoder.final_dim, self.online_encoder.final_dim),
                           # nn.ReLU(), nn.Linear(self.online_encoder.final_dim, self.online_encoder.final_dim))
        # uncomment if you do more tasks
        # self.stem_classifiers = nn.Linear(self.online_encoder.final_dim, 2)

        # self.immune_classifiers = nn.Linear(self.online_encoder.final_dim, 2)

        self.apply(initialize_weights)

    def forward(self, x, return_attn=False,no_norm=False):
        x = self.patch_to_emb(x) # n*512
        x = self.dp(x)
        
        # feature re-embedding
        features= self.online_encoder(x)

        # features = self.feature_extractor(x)

        patch_level_logits_cell_type, _ = self.cell_type_classifiers(features)
        # patch_level_logits_cell_type = self.cell_type_classifiers(x)

        cell_type_prob = F.softmax(patch_level_logits_cell_type, dim = 1).mean(dim=0)

        # patch_level_logits_stem = self.stem_classifiers(x)
        # patch_level_logits_immune = self.immune_classifiers(x)
        # stem_prob = F.softmax(patch_level_logits_stem, dim = 1).mean(dim=0)
        # immune_prob = F.softmax(patch_level_logits_immune, dim = 1).mean(dim=0)
       
        # if return_attn:
        # # feature aggregation
        #     x,a = self.pool_fn(x, return_attn=True,no_norm=no_norm)
         
        # else:
        #     x = self.pool_fn(x)
        
        # prediction
        # logits = self.predictor(x)
        
        # results_dict = {
        #     'stem_logits': patch_level_logits_stem, 
        #     'immune_logits': patch_level_logits_immune,
        #     'cell_type_logits': patch_level_logits_cell_type,
        #     'cell_type_prob': cell_type_prob, 
        #     'stem_prob': stem_prob,
        #     'immune_prob': immune_prob
        # }   
        results_dict = {
            'cell_type_logits': patch_level_logits_cell_type,
            'cell_type_prob': cell_type_prob,
            'features': features
        }

      
        return results_dict
        
        
if __name__ == "__main__":
    x = torch.rand(62652,1024)
    x_rrt = torch.rand(62652,512)

    # epeg_k，crmsa_k are the primary hyper-para, you can set crmsa_heads, all_shortcut and crmsa_mlp if you want.
    # C16-R50: input_dim=1024,epeg_k=15,crmsa_k=1,crmsa_heads=8,all_shortcut=True
    # C16-PLIP: input_dim=512,epeg_k=9,crmsa_k=3,crmsa_heads=8,all_shortcut=True
    # TCGA-LUAD&LUSC-R50: input_dim=1024,epeg_k=21,crmsa_k=5,crmsa_heads=8
    # TCGA-LUAD&LUSC-PLIP: input_dim=512,epeg_k=13,crmsa_k=3,crmsa_heads=1,all_shortcut=True,crmsa_mlp=True
    # TCGA-BRCA-R50:input_dim=1024,epeg_k=17,crmsa_k=3,crmsa_heads=1
    # TCGA-BRCA-PLIP: input_dim=512,epeg_k=15,crmsa_k=1,crmsa_heads=8,all_shortcut=True

    # rrt+abmil
    rrt_mil = RRTMIL(n_classes=3,epeg_k=15,crmsa_k=3)
    x = rrt_mil(x)  # 1,N,D -> 1,C
    
    # rrt. you should put the rrt_enc before aggregation module, after fc and dp
    # x_rrt = fc(x_rrt) # 1,N,1024 -> 1,N,512
    # x_rrt = dropout(x_rrt)
    rrt = RRTEncoder(mlp_dim=512,epeg_k=15,crmsa_k=3) 
    x_rrt = rrt(x_rrt) # 1,N,512 -> 1,N,512
    # x_rrt = mil_model(x_rrt) # 1,N,512 -> 1,N,C

    print(x.size())
    print(x_rrt.size())
