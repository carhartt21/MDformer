import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops.roi_align as roi_align # Instance Aware
from einops import repeat, rearrange

from . import vit
from . import blocks
import logging

# Enable debug logging
# logging.basicConfig(level=logging.DEBUG)

class PreInstanceNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = blocks.AdaptiveInstanceNorm2d(dim,eps=1e-06)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class AdaIn_Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., vis=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.vis = vis
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreInstanceNorm(dim, vit.Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout, vis = vis)),
                PreInstanceNorm(dim, vit.FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
        self.hooks = []
        self.features = None

    def set_hooks(self, hooks):
        self.hooks = hooks

    def forward(self, x):
        i = 0
        w = []
        ll = []
        for attn, ff in self.layers:
            _x, _w = attn(x)
            x += _x 
            if self.vis:
                w.append(_w)      
            x = ff(x) + x
            if i in self.hooks:
                ll.append(x)
            i += 1

        self.features = tuple(ll)
        return x, w


class Transformer_Aggregator(nn.Module):
    # TODO: Add support for multiple layers
    
    def __init__(self, input_size: int = 88, patch_size: int = 8, patch_embed_C: int = 1024, sem_embed_C: int = 64, feat_C: int = 256, depth: int = 6, heads: int = 4, mlp_dim: int =4096, num_sem_classes: int = 16, num_classes: int = 16, vis: bool = False):
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
            vis: visualize attention maps
        '''
        super(Transformer_Aggregator, self).__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.patch_embed_C = patch_embed_C
        self.sem_embed_C = sem_embed_C
        self.vis = vis

        # Patch Embedding
        self.patch_embed = blocks.PatchEmbedding(
            input_size=input_size, patch_size=patch_size, in_chans=feat_C, embed_dim=self.patch_embed_C)
        self.box_embed = blocks.PatchEmbedding(
            input_size=patch_size, patch_size=patch_size, in_chans=feat_C, embed_dim=self.patch_embed_C)
        
        # Semantic Embedding
        self.sem_embed = blocks.SemanticEmbedding(num_sem_classes=num_sem_classes, embed_dim=self.sem_embed_C)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.patch_embed_C+self.sem_embed_C))

        # Positional Embedding
        self.total_embed_C = patch_embed_C + sem_embed_C
        assert self.total_embed_C % 4 == 0, 'The number of total embedding channels must be divisible by 4.'
        self.pos_embed_x = blocks.PositionalEncoding(input_dim=self.total_embed_C//4, dropout=0., max_len=input_size)
        self.pos_embed_y = blocks.PositionalEncoding(input_dim=self.total_embed_C//4, dropout=0., max_len=input_size)
        self.pos_embed_h = blocks.PositionalEncoding(input_dim=self.total_embed_C//4, dropout=0., max_len=input_size)
        self.pos_embed_w = blocks.PositionalEncoding(input_dim=self.total_embed_C//4, dropout=0., max_len=input_size)
    
        self.block_pos_embed_x = blocks.PositionalEncoding(input_dim=self.patch_embed_C//4, dropout=0., max_len=input_size)
        self.block_pos_embed_y = blocks.PositionalEncoding(input_dim=self.patch_embed_C//4, dropout=0., max_len=input_size)
        self.block_pos_embed_h = blocks.PositionalEncoding(input_dim=self.patch_embed_C//4, dropout=0., max_len=input_size)
        self.block_pos_embed_w = blocks.PositionalEncoding(input_dim=self.patch_embed_C//4, dropout=0., max_len=input_size)
    

        # Concatenate positional embeddings
        pos_embed = torch.cat((
            self.pos_embed_x()[..., None, :].repeat(1, 1, input_size, 1),
            self.pos_embed_y()[:, None].repeat(1, input_size, 1, 1),
            self.pos_embed_w()[:, (8 - 1)][:, None, None].repeat(1, input_size, input_size, 1),    
            self.pos_embed_h()[:, (8 - 1)][:, None, None].repeat(1, input_size, input_size, 1),
            ), dim=3)
        # Resize positional embedding to match the size of the patch embedding
        self.pos_embed = F.interpolate(pos_embed.permute(0, 3, 1, 2), size=((input_size//patch_size), input_size//patch_size), mode='bilinear').flatten(2).transpose(-1, -2)
        
        # Add empty embedding for cls_token
        cls_embed = torch.zeros(1, 1, self.total_embed_C)
        self.pos_embed = torch.cat((cls_embed, self.pos_embed), dim=1)

        # Transformer
        self.transformer = AdaIn_Transformer(dim=self.total_embed_C, depth=depth, heads=heads, dim_head=feat_C, mlp_dim=mlp_dim, dropout=0., vis= self.vis)

    def extract_box_feature(self, out, box, n_bbox):
        '''
        Extracts features from bounding boxes using ROI Align.
        Args:
            out: Output feature map from the previous layer
            box: Bounding box coordinates
            n_bbox: Number of bounding boxes
        Returns:
            aligned_out: Aligned features from the bounding boxes
        '''
        b, c, h, w = out.shape
        batch_index = torch.arange(0.0, b).repeat(n_bbox).view(n_bbox, -1).transpose(0,1).flatten(0,1).to(out.device)
        roi_bbox_info = box.view(-1,5).to(out.device) 

        roi_info = torch.stack((batch_index, \
                            roi_bbox_info[:, 1] * w, \
                            roi_bbox_info[:, 2] * h, \
                            roi_bbox_info[:, 3] * w, \
                            roi_bbox_info[:, 4] * h), dim = 1).to(out.device)
        aligned_out = roi_align(out, roi_info, 8) 

        aligned_out.view(b, n_bbox, c, 8, 8)[torch.where(box[:,:,0] == -1)] = 0
        aligned_out.view(-1, c, 8, 8)        

        return aligned_out
    
    def add_box(self, out, box, bbox_info, n_bbox, pos_embed=None): 
            """
            Adds a box to the output tensor.

            Args:
                out (torch.Tensor): The output tensor.
                box (torch.Tensor): The box tensor.
                bbox_info (torch.Tensor): The bounding box information tensor.
                n_bbox (int): The number of bounding boxes.
                pos_embed (torch.Tensor, optional): The positional embedding tensor. Defaults to None.

            Returns:
                torch.Tensor: The updated output tensor with the added box.
            """
            
            b = out.shape[0]
            box = self.box_embed(box).squeeze().view(b, n_bbox, -1) 
            
            x_coord = (bbox_info[..., 1::2].mean(dim=2) * (self.input_size - 1)).long() 
            y_coord = (bbox_info[..., 2::2].mean(dim=2) * (self.input_size - 1)).long() 
            w = ((bbox_info[..., 3] - bbox_info[..., 1]) * (self.input_size - 1)).long()
            h = ((bbox_info[..., 4] - bbox_info[..., 2]) * (self.input_size - 1)).long()
            

            box_pos_embed = torch.cat((
                self.block_pos_embed_x()[..., None, :].repeat(1, 1, self.input_size, 1), 
                self.block_pos_embed_y()[:, None].repeat(1, self.input_size, 1, 1), 
                ), dim=3).squeeze()

            box += torch.cat((
                    box_pos_embed[y_coord, x_coord], box_pos_embed[h, w] 
                    ), dim=2)
            box = self.sem_embed(box, bbox_info[:,:,0])
            added_out = torch.cat((out, box), dim=1)
            
            return added_out
    
    def resize_pos_embed(self, h, w, start_index=1):
            """
            Resize the positional embeddings based on the new image size.

            Args:
                h (int): The new height
                w (int): The new width
                start_index (int, optional): The index to start slicing the positional embeddings. Defaults to 1.
            """
            
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

    def forward(self, x, sem_embed=True, sem_labels=None, bbox_info=None, n_bbox=-1):
        """
        Forward pass of the Transformer Aggregator module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
            sem_embed (bool): Flag indicating whether to apply semantic embeddings. Default is True.
            sem_labels (torch.Tensor): Semantic labels tensor of shape (batch_size, num_classes).
                Required when sem_embed is True.
            bbox_info (torch.Tensor): Bounding box information tensor of shape (batch_size, num_boxes, 4).
                Default is None.
            n_bbox (int): Number of bounding boxes. Default is -1.

        Returns:
            aggregated_feat (torch.Tensor): Aggregated feature tensor of shape (batch_size, num_patches, hidden_dim).
            aggregated_box (torch.Tensor): Aggregated box tensor of shape (batch_size, num_boxes, hidden_dim).
                None if n_bbox <= 0.
        """
        # Apply the patch embedding
        patch_embed_x = self.patch_embed(x)

        # Extract class tokens
        cls_token = repeat(self.cls_token, '1 n d -> b n d', b=(x.shape[0]))

        # Add semantic embeddings
        # if sem_embed:
        if sem_labels is None:
            # raise ValueError("Semantic labels are required when sem_embed is True.")
            sem_labels = torch.zeros(patch_embed_x.shape[0], patch_embed_x.shape[1], dtype=torch.long).to(x.device)
        # Add semantic embeddings to 'patch_embed_x'
        # if bbox_info is not None:
        #     sem_labels = torch.cat([sem_labels, bbox_info[:,:,0]], dim=1)
        patch_embed_x = self.sem_embed(patch_embed_x, sem_labels)        

        patch_embed_x = torch.cat((cls_token, patch_embed_x), dim=1)

        patch_embed_x = patch_embed_x + self.pos_embed.to(x.device)

        # Add box embeddings
        if bbox_info is not None:
            box_feat = self.extract_box_feature(x, bbox_info, n_bbox)
            patch_embed_x = self.add_box(patch_embed_x, box_feat, bbox_info, n_bbox)
        
        # Concatenate class tokens and embedded features
        # Add positional embeddings except for the box embeddings

        # Pass the modified 'x' through the transformer layer
        out, weights = self.transformer(patch_embed_x)

        # Extract the aggregated features and boxes
        if n_bbox > 0:
            aggregated_feat, aggregated_box = out[:, :-n_bbox, :], out[:, -n_bbox:, :]
        else:
        # If n_bbox <= 0, return only the aggregated features
            aggregated_feat, aggregated_box = out, None

        return aggregated_feat, aggregated_box, weights
        
        
