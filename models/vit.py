import torch
from torch import nn
import torch.nn.functional as F


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim,eps=1e-06)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., vis=False):
        # Args:
        #   dim: input dimension 
        #   heads: number of attention heads
        #   dim_head: dimension of each head
        #   dropout: dropout rate
        # 

        super().__init__()
        self.vis = vis
        assert dim % heads == 0

        # inner dimension of the model
        inner_dim = dim_head *  heads

        # project output if necessary
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = True)

        # use identity function if project_out is False
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        attn = self.attend(dots)
        if self.vis:
            weights = attn
        else:
            weights = None
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), weights

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
        self.hooks = []
        self.features = None

    def set_hooks(self, hooks):
        self.hooks = hooks
     
    # def forward(self, x):
    #     for attn, ff in self.layers:
    #         x = attn(x) + x
    #         x = ff(x) + x
    #     return x
    
    def forward(self, x, mask=None):
        i = 0
        ll = []
        for attn, ff in self.layers:
            if mask == None:
                x = attn(x) + x
            else:
                x = attn.fn(attn.norm(x),mask) + x
                
            x = ff(x) + x
            if i in self.hooks:
                ll.append(x)
            i += 1
        self.features = tuple(ll)
    
        return x

        
