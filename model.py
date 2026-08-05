from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from layers import GraphConvolution, DistanceAdj


def check_tensor(tensor, name):
    if torch.isnan(tensor).any():
        raise ValueError(f"Tensor {name} contains NaN")
    if torch.isinf(tensor).any():
        raise ValueError(f"Tensor {name} contains Inf")

class LayerNorm(nn.LayerNorm):

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, padding_mask: torch.Tensor):
        padding_mask = padding_mask.to(dtype=bool, device=x.device) if padding_mask is not None else None
        self.attn_mask = self.attn_mask.to(device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, key_padding_mask=padding_mask, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        x, padding_mask = x
        x = x + self.attention(self.ln_1(x), padding_mask)
        x = x + self.mlp(self.ln_2(x))
        return (x, padding_mask)



class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class WVAD_Model(nn.Module):
    def __init__(self,
                 embed_dim : int = 512,
                 visual_length: int = 256,
                 visual_head: int = 1,
                 visual_layers: int = 2,
                 attn_window: int = 8,
                 devices: str = "cuda"
                 ):
        super().__init__()

        self.visual_length = visual_length
        self.embed_dim = embed_dim
        self.attn_window = attn_window
        self.device = devices

        self.down_proj = nn.Sequential(OrderedDict([
            ("p_fc", nn.Linear(2816, embed_dim)),
            ("gelu", QuickGELU())
        ]))

        # Transformer需要改造
        self.temporal = Transformer(
            width=embed_dim,
            layers=visual_layers,
            heads=visual_head,
            attn_mask=self.build_attention_mask(self.attn_window)
        )

        hidden_dim = int(embed_dim / 2)
        self.gc1 = GraphConvolution(embed_dim, hidden_dim, residual=True)
        self.gc2 = GraphConvolution(hidden_dim, hidden_dim, residual=True)
        self.gc3 = GraphConvolution(embed_dim, hidden_dim, residual=True)
        self.gc4 = GraphConvolution(hidden_dim, hidden_dim, residual=True)

        self.disAdj = DistanceAdj()
        self.linear = nn.Linear(embed_dim, embed_dim)
        self.gelu = QuickGELU()

        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(embed_dim, embed_dim * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(embed_dim * 4, embed_dim))
        ]))
        self.classifier = nn.Linear(embed_dim, 1)

        self.frame_position_embeddings = nn.Embedding(visual_length, 2048 + 768).to(devices)

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.frame_position_embeddings.weight, std=0.01)

    def build_attention_mask(self, attn_window):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.visual_length, self.visual_length)
        mask.fill_(float('-inf'))
        for i in range(int(self.visual_length / attn_window) + 1):
            if (i + 1) * attn_window < self.visual_length:
                mask[i * attn_window: (i + 1) * attn_window, i * attn_window: (i + 1) * attn_window] = 0
            else:
                mask[i * attn_window: self.visual_length, i * attn_window: self.visual_length] = 0

        return mask

    def adj4(self, x, seq_len):
        soft = nn.Softmax(1)
        x2 = x.matmul(x.permute(0, 2, 1)) # B*T*T
        x_norm = torch.norm(x, p=2, dim=2, keepdim=True)  # B*T*1
        x_norm_x = x_norm.matmul(x_norm.permute(0, 2, 1))
        x2 = x2/(x_norm_x+1e-20)
        output = torch.zeros_like(x2)
        if seq_len is None:
            for i in range(x.shape[0]):
                tmp = x2[i]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.7, 0)
                adj2 = soft(adj2)
                output[i] = adj2
        else:
            for i in range(len(seq_len)):
                tmp = x2[i, :seq_len[i], :seq_len[i]]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.7, 0)
                adj2 = soft(adj2)
                output[i, :seq_len[i], :seq_len[i]] = adj2

        return output

    def encode_video(self, clip_feature, I3D_feature, lengths):

        clip_feature = clip_feature.to(torch.float)
        I3D_feature = I3D_feature.to(torch.float)

        features = torch.cat([clip_feature, I3D_feature], dim=-1).to(self.device)
        
        position_ids = torch.arange(self.visual_length, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand(features.shape[0], -1) # expand to [B, T]
        frame_position_embeddings = self.frame_position_embeddings(position_ids)
        frame_position_embeddings = frame_position_embeddings.permute(1, 0, 2)
        
        features = features.permute(1, 0, 2) + frame_position_embeddings

        x = self.down_proj(features)

        x, _ = self.temporal((x, None))
        x = x.permute(1, 0, 2)

        adj = self.adj4(x, lengths)
        disadj = self.disAdj(x.shape[0], x.shape[1])
        x1_h = self.gelu(self.gc1(x, adj))
        x2_h = self.gelu(self.gc3(x, disadj))

        x1 = self.gelu(self.gc2(x1_h, adj))
        x2 = self.gelu(self.gc4(x2_h, disadj))

        x = torch.cat((x1, x2), 2)
        x = self.linear(x)

        return x
    def forward(self, clip_feature, I3D_feature, lengths):
        visual_features = self.encode_video(clip_feature, I3D_feature, lengths)

        logits = self.classifier(visual_features + self.mlp(visual_features))

        return logits