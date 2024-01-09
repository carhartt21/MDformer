import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops.roi_align as roi_align # Instance Aware
from einops.layers.torch import Rearrange
from einops import repeat, rearrange

from . import vit
from . import blocks

class PreInstanceNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = blocks.AdaptiveInstanceNorm2d(dim,eps=1e-06)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class AdaIn_Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreInstanceNorm(dim, vit.Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreInstanceNorm(dim, vit.FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
        self.hooks = []
        self.features = None

    def set_hooks(self, hooks):
        self.hooks = hooks

    def forward(self, x):
        i = 0
        ll = []
        for attn, ff in self.layers:
            x = attn(x) + x      
            x = ff(x) + x
            if i in self.hooks:
                ll.append(x)
            i += 1

        self.features = tuple(ll)
        return x

class Transformer_Aggregator(nn.Module):
    # TODO: Add support for multiple layers
    def __init__(self, input_size: int = 88, patch_size: int = 8, embed_C: int = 1024, feat_C: int = 256, depth: int = 6, heads: int = 4, mlp_dim: int =4096):
        '''
        Transformer Aggregator
        Args:
            input_size: image size at transformer input
            patch_size: patch size for patch embedding
            embed_C: embedding channels
            feat_C: feature channels
            depth: dimension of transformer
            heads: heads of transformer
            mlp_dim: mlp dimension
        '''
        super(Transformer_Aggregator, self).__init__()
        self.input_size = input_size
        self.patch_size = patch_size

        self.patch_embed = blocks.PatchEmbedding(
            input_size=input_size, patch_size=patch_size, in_chans=feat_C, embed_dim=embed_C)
        self.box_embed = blocks.PatchEmbedding(
            input_size=patch_size, patch_size=patch_size, in_chans=feat_C, embed_dim=embed_C)
        
        # semantic embedding
        # self.seg_embed = blocks.SemanticEmbedding(input_size=input_size, patch_size=patch_size)

        self.pos_embed_x = blocks.PositionalEncoding(input_dim=embed_C//4, dropout=0., max_len=input_size)
        self.pos_embed_y = blocks.PositionalEncoding(input_dim=embed_C//4, dropout=0., max_len=input_size)
        self.pos_embed_h = blocks.PositionalEncoding(input_dim=embed_C//4, dropout=0., max_len=input_size)
        self.pos_embed_w = blocks.PositionalEncoding(input_dim=embed_C//4, dropout=0., max_len=input_size)


        # position embedding for box with center coordinates x, y, width w and height h
        pos_embed = torch.cat((
                self.pos_embed_x()[..., None, :].repeat(1, 1, input_size, 1), # 
                self.pos_embed_y()[:, None].repeat(1, input_size, 1, 1), #
                self.pos_embed_w()[:, (8 - 1)][:, None, None].repeat(1, input_size, input_size, 1),    
                self.pos_embed_h()[:, (8 - 1)][:, None, None].repeat(1, input_size, input_size, 1),
                ), dim=3)
        self.pos_embed = F.interpolate(pos_embed.permute(0, 3, 1, 2), size=(input_size//patch_size, input_size//patch_size), mode='bilinear').flatten(2).transpose(-1, -2)

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_C))

        self.transformer = AdaIn_Transformer(dim=embed_C, depth=depth, heads=heads, dim_head=feat_C, mlp_dim=mlp_dim, dropout=0.)

    def extract_box_feature(self, out, box, n_bbox):
        b, c, h, w = out.shape
        batch_index = torch.arange(0.0, b).repeat(n_bbox).view(n_bbox, -1).transpose(0,1).flatten(0,1).to(out.device)
        roi_box_info = box.view(-1,5).to(out.device) 

        roi_info = torch.stack((batch_index, \
                            roi_box_info[:, 1] * w, \
                            roi_box_info[:, 2] * h, \
                            roi_box_info[:, 3] * w, \
                            roi_box_info[:, 4] * h), dim = 1).to(out.device)
        aligned_out = roi_align(out, roi_info, 8) 

        aligned_out.view(b, n_bbox, c, 8, 8)[torch.where(box[:,:,0] == -1)] = 0
        aligned_out.view(-1, c, 8, 8)        

        return aligned_out
    
    def add_box(self, out, box, box_info, n_bbox, pos_embed=None): 
        b = out.shape[0]
        box = self.box_embed(box).squeeze().view(b, n_bbox, -1) 
        
        x_coord = (box_info[..., 1::2].mean(dim=2) * (self.input_size - 1)).long() 
        y_coord = (box_info[..., 2::2].mean(dim=2) * (self.input_size - 1)).long() 
        w = ((box_info[..., 3] - box_info[..., 1]) * (self.input_size - 1)).long()
        h = ((box_info[..., 4] - box_info[..., 2]) * (self.input_size - 1)).long()
        
        box_pos_embed = torch.cat((
            self.pos_embed_x()[..., None, :].repeat(1, 1, self.input_size, 1), 
            self.pos_embed_y()[:, None].repeat(1, self.input_size, 1, 1), 
            ), dim=3).squeeze()

        box += torch.cat((
                box_pos_embed[y_coord, x_coord], box_pos_embed[h, w] 
                ), dim=2)

        added_out = torch.cat((out, box), dim=1) 
        
        return added_out
    
    def resize_pos_embed(self, h, w, start_index=1):
        old_h, old_w = self.__image_size
        self.__image_size = (h,w)
        pw, ph = self.get_patch_size()
        gs_w = w//pw
        gs_h = h//ph
        
        posemb_tok, posemb_grid = (
            self.pos_embedding[:, : start_index],
            self.pos_embedding[0, start_index :],
        )

                
        gs_old_w = old_w // pw
        gs_old_h = old_h // ph

        posemb_grid = posemb_grid.reshape(1, gs_old_h, gs_old_w, -1).permute(0, 3, 1, 2)
        posemb_grid = F.interpolate(posemb_grid, size=(gs_h, gs_w), mode="bilinear")
        posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_h * gs_w, -1)

        self.pos_embedding = nn.Parameter(torch.cat([posemb_tok, posemb_grid], dim=1))

    def forward(self, x, semantic_embed = False, box_info=None, n_bbox=-1):
        if semantic_embed:
            embed_x = self.patch_embed(x) + self.pos_embed.to(x.device) + self.sem_embed(x)
        else:
            embed_x = self.patch_embed(x) + self.pos_embed.to(x.device)
        
        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b = x.shape[0])
        # print("shapes: ", cls_tokens.shape, embed_x.shape)
        # x = torch.cat((cls_tokens, embed_x), dim=1)

        if box_info != None:
            box_feat = self.extract_box_feature(x, box_info, n_bbox)
            embed_x = self.add_box(embed_x, box_feat, box_info, n_bbox)
        out = self.transformer(embed_x)
        if n_bbox > 0:
            aggregated_feat, aggregated_box = out[:,:-n_bbox,:], out[:, -n_bbox:, :]
        else:
            aggregated_feat, aggregated_box = out, None 
        
        return aggregated_feat, aggregated_box
        
        
