import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops.roi_align as roi_align  # Instance Aware
from einops import repeat, rearrange
from torchvision import transforms

from . import swin
from . import vit
from . import blocks
from utils import _ntuple
import logging
import math
from util.custom_transforms import SegMaskToPatches

# Enable debug logging
# logging.basicConfig(level=logging.DEBUG)


class PreInstanceNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = blocks.AdaptiveInstanceNorm2d(dim, eps=1e-06)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class AdaIN_Swin_Transformer(nn.Module):
    def __init__(
        self,
        swin_cfg, # swin config
        input_size=(128, 128),
        patch_size=(4, 4),
        in_chans=3,
        num_classes=1000,
        dim=96,
        # depths=[2, 2, 6, 2],
        # num_heads=[3, 6, 12, 24],
        # window_size=7,
        # mlp_ratio=4.0,
        # qkv_bias=True,
        # drop_rate=0.0,
        # attn_drop_rate=0.0,
        drop_path_rate=0.1,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        pretrained_window_sizes=[0, 0, 0, 0],
        vis=False,
        **kwargs
    ):
        super().__init__()
        # self.num_classes = num_classes
        self.depths = swin_cfg.depths
        self.num_layers = len(self.depths)
        self.embed_dim = dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(dim * 2 ** (self.num_layers - 1))
        self.layers = nn.ModuleList([])
        self.vis = vis
        # self.pos_drop = nn.Dropout(p=swin_cfg.drop_rate)

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))
        ]  # stochastic depth decay rule

        patches_resolution = [
            input_size[0] // patch_size[0],
            input_size[1] // patch_size[1],
        ]

        # build layers
        for i_layer in range(self.num_layers):
            layer = swin.BasicLayer(
                dim=int(dim * 2**i_layer),
                input_resolution=(
                    patches_resolution[0] // (2**i_layer),
                    patches_resolution[1] // (2**i_layer),
                ),
                depth=self.depths[i_layer],
                num_heads=swin_cfg.num_heads[i_layer],
                window_size=swin_cfg.window_size,
                mlp_ratio=swin_cfg.mlp_ratio,
                qkv_bias=swin_cfg.qkv_bias,
                drop=swin_cfg.drop_path_rate,
                attn_drop=swin_cfg.attn_drop_rate,
                drop_path=dpr[sum(swin_cfg.depths[:i_layer]) : sum(self.depths[: i_layer + 1])],
                norm_layer=blocks.AdaptiveInstanceNorm2d,
                downsample=swin.PatchMerging
                if (i_layer < self.num_layers - 1)
                else None,
                pretrained_window_size=pretrained_window_sizes[i_layer],
            )
            self.layers.append(layer)

    def forward(self, x):
        # x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        # x = self.norm(x)  # B L C
        # x = self.avgpool(x.transpose(1, 2))  # B C 1
        # x = torch.flatten(x, 1)
        # B x patch_size*patch_size x dim
        # 8 x 16 x 8448
        return x, []


class AdaIn_Transformer(nn.Module):
    def __init__(self, vit_cfg, dim, depth, vis=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.vis = vis
        depth = vit_cfg.depth
        
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreInstanceNorm(
                            dim,
                            vit.Attention(
                                dim,
                                heads=vit_cfg.heads,
                                dim_head=depth,
                                dropout=vit_cfg.dropout,
                                vis=vis,
                            ),
                        ),
                        PreInstanceNorm(
                            dim, vit.FeedForward(dim=dim, hidden_dim=vit_cfg.mlp_dim, dropout=vit_cfg.dropout)
                        ),
                    ]
                )
            )
        self.hooks = []
        self.features = None

    def set_hooks(self, hooks):
        self.hooks = hooks

    def forward(self, x):
        i = 0
        weights = []
        ll = []
        for attention, feed_forward in self.layers:
            _x, _w = attention(x)
            x += _x
            if self.vis:
                weights.append(_w)
            x = feed_forward(x) + x
            if i in self.hooks:
                ll.append(x)
            i += 1

        self.features = tuple(ll)
        return x, weights


class Transformer(nn.Module):
    def __init__(
        self,
        model_cfg,
        num_sem_classes: int = 16,
        vis: bool = False,
    ):
        super(Transformer, self).__init__()
        self.transformer_type = model_cfg.transformer_type
        self.vis = vis
        self.total_embed_C = model_cfg.patch_embed_dim + model_cfg.sem_embed_dim

        # Transformer
        if self.transformer_type == "vit":
            self.embedding = Embedding(
                input_size=model_cfg.img_size[0]//2**model_cfg.n_downsampling,
                in_chans=model_cfg.n_generator_filters * 2**model_cfg.n_downsampling,
                patch_size=model_cfg.patch_size,
                patch_embed_C=model_cfg.patch_embed_dim,
                sem_embed_C=model_cfg.sem_embed_dim,
                num_sem_classes=num_sem_classes,
                vis=self.vis,
            )
            self.transformer = AdaIn_Transformer(
                vit_cfg = model_cfg.VIT,
                dropout=0.0,
                vis=self.vis,
            )
        elif self.transformer_type == "swin":
            self.embedding = Embedding(
                input_size=model_cfg.img_size[0]//2**model_cfg.n_downsampling,
                in_chans=model_cfg.n_generator_filters * 2**model_cfg.n_downsampling,
                patch_size=model_cfg.SWIN.patch_size,
                patch_embed_C=model_cfg.patch_embed_dim,
                sem_embed_C=model_cfg.sem_embed_dim,
                num_sem_classes=num_sem_classes,
                vis=self.vis,
            )            
            self.transformer = AdaIN_Swin_Transformer(
                dim=self.total_embed_C,
                input_size=[s//2**model_cfg.n_downsampling for s in model_cfg.img_size],
                patch_size=[model_cfg.SWIN.patch_size, model_cfg.SWIN.patch_size],
                swin_cfg=model_cfg.SWIN,
                depths=model_cfg.SWIN.depths,
                heads=model_cfg.SWIN.num_heads,
                drop_path_rate=model_cfg.SWIN.drop_path_rate,
                vis=self.vis,
            )
        else:
            raise ValueError("Transformer type not supported.")

    def forward(self, x, sem_embed=True, sem_labels=None, bbox_info=None, n_bbox=-1):
        embed_x = self.embedding(x, sem_embed, sem_labels, bbox_info, n_bbox)
        # Pass the modified 'x' through the transformer layer
        out, weights = self.transformer(embed_x)

        # Extract the aggregated features and boxes
        if n_bbox > 0:
            aggregated_feat, aggregated_box = out[:, :-n_bbox, :], out[:, -n_bbox:, :]
        else:
            # If n_bbox <= 0, return only the aggregated features
            aggregated_feat, aggregated_box = out, None

        return aggregated_feat, aggregated_box, weights

    def apply_adain_params(self, adain_params):
        # assign the adain_params to the AdaIN layers in model
        for layer in self.transformer.layers:
            if self.transformer_type == "vit":
                for _layer in layer:
                    if isinstance(_layer, blocks.AdaptiveInstanceNorm2):
                        mean = adain_params[:, :_layer.num_features]
                        std = adain_params[:, _layer.num_features:2*_layer.num_features]
                        _layer.bias = mean.contiguous().view(-1)
                        _layer.weight = std.contiguous().view(-1)
                        # if adain_params.size(1) > 2*_layer.num_features:
                        #     adain_params = adain_params[:, 2*_layer.num_features:]
            elif self.transformer_type == "swin":
                for swin_block in layer.blocks:
                    m = swin_block.norm1
                    mean = adain_params[:, :m.num_features]
                    std = adain_params[:, m.num_features:2*m.num_features]
                    m.bias = mean.contiguous().view(-1)
                    m.weight = std.contiguous().view(-1)
                    # if adain_params.size(1) > 2*m.num_features:
                    #     adain_params = adain_params[:, 2*m.num_features:]
                    n = swin_block.norm2
                    mean = adain_params[:, :n.num_features]
                    std = adain_params[:, n.num_features:2*n.num_features]
                    n.bias = mean.contiguous().view(-1)
                    n.weight = std.contiguous().view(-1)
                    # if adain_params.size(1) > 2*n.num_features:
                    #     adain_params = adain_params[:, 2*n.num_features:]                    

class Embedding(nn.Module):
    # TODO: Add support for multiple layers

    def __init__(
        self,
        input_size: int = 88,
        patch_size: int = 8,
        patch_embed_C: int = 1024,
        sem_embed_C: int = 64,
        in_chans: int = 256,
        num_sem_classes: int = 16,
        vis: bool = False,
    ):
        """
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
        """
        super(Embedding, self).__init__()
        logging.info(f"Embedding input_size: {input_size}, patch_size: {patch_size}, patch_embed_C: {patch_embed_C}, sem_embed_C: {sem_embed_C}")
        self.input_size = input_size
        self.patch_size = patch_size
        self.patch_embed_C = patch_embed_C
        self.sem_embed_C = sem_embed_C
        self.vis = vis
        stem = True if patch_size == 8 else False

        # Patch Embedding
        self.patch_embed = PatchEmbedding(
            input_size=input_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=self.patch_embed_C,
            STEM=stem,
        )
        self.box_embed = PatchEmbedding(
            input_size=patch_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=self.patch_embed_C,
            STEM=stem,
        )

        # Semantic Embedding
        self.sem_embed = SemanticEmbedding(
            num_sem_classes=num_sem_classes, embed_dim=self.sem_embed_C, patch_size=patch_size, content_dim=input_size
        )
        # self.cls_token = nn.Parameter(torch.randn(1, 1, self.patch_embed_C+self.sem_embed_C))

        # Positional Embedding
        self.total_embed_C = patch_embed_C + sem_embed_C
        assert (
            self.total_embed_C % 4 == 0
        ), "The number of total embedding channels must be divisible by 4."
        self.pos_embed_x = PositionalEncoding(
            input_dim=self.total_embed_C // 4, dropout=0.0, max_len=input_size
        )
        self.pos_embed_y = PositionalEncoding(
            input_dim=self.total_embed_C // 4, dropout=0.0, max_len=input_size
        )
        self.pos_embed_h = PositionalEncoding(
            input_dim=self.total_embed_C // 4, dropout=0.0, max_len=input_size
        )
        self.pos_embed_w = PositionalEncoding(
            input_dim=self.total_embed_C // 4, dropout=0.0, max_len=input_size
        )

        self.block_pos_embed_x = PositionalEncoding(
            input_dim=self.patch_embed_C // 4, dropout=0.0, max_len=input_size
        )
        self.block_pos_embed_y = PositionalEncoding(
            input_dim=self.patch_embed_C // 4, dropout=0.0, max_len=input_size
        )
        self.block_pos_embed_h = PositionalEncoding(
            input_dim=self.patch_embed_C // 4, dropout=0.0, max_len=input_size
        )
        self.block_pos_embed_w = PositionalEncoding(
            input_dim=self.patch_embed_C // 4, dropout=0.0, max_len=input_size
        )

        # Concatenate positional embeddings
        pos_embed = torch.cat(
            (
                self.pos_embed_x()[..., None, :].repeat(1, 1, input_size, 1),
                self.pos_embed_y()[:, None].repeat(1, input_size, 1, 1),
                self.pos_embed_w()[:, (8 - 1)][:, None, None].repeat(
                    1, input_size, input_size, 1
                ),
                self.pos_embed_h()[:, (8 - 1)][:, None, None].repeat(
                    1, input_size, input_size, 1
                ),
            ),
            dim=3,
        )
        # Resize positional embedding to match the size of the patch embedding
        self.pos_embed = (
            F.interpolate(
                pos_embed.permute(0, 3, 1, 2),
                size=((input_size // patch_size), input_size // patch_size),
                mode="bilinear",
            )
            .flatten(2)
            .transpose(-1, -2)
        )

    def extract_box_feature(self, out, box, n_bbox, patch_size=8):
        """
        Extracts features from bounding boxes using ROI Align.
        Args:
            out: Output feature map from the previous layer
            box: Bounding box coordinates
            n_bbox: Number of bounding boxes
        Returns:
            aligned_out: Aligned features from the bounding boxes
        """
        b, c, h, w = out.shape
        batch_index = (
            torch.arange(0.0, b)
            .repeat(n_bbox)
            .view(n_bbox, -1)
            .transpose(0, 1)
            .flatten(0, 1)
            .to(out.device)
        )
        roi_bbox_info = box.view(-1, 5).to(out.device)

        roi_info = torch.stack(
            (
                batch_index,
                roi_bbox_info[:, 1] * w,
                roi_bbox_info[:, 2] * h,
                roi_bbox_info[:, 3] * w,
                roi_bbox_info[:, 4] * h,
            ),
            dim=1,
        ).to(out.device)
        aligned_out = roi_align(out, roi_info, patch_size)

        aligned_out.view(b, n_bbox, c, patch_size, patch_size)[torch.where(box[:, :, 0] == -1)] = 0
        aligned_out.view(-1, c, patch_size, patch_size)

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

        box_pos_embed = torch.cat(
            (
                self.block_pos_embed_x()[..., None, :].repeat(1, 1, self.input_size, 1),
                self.block_pos_embed_y()[:, None].repeat(1, self.input_size, 1, 1),
            ),
            dim=3,
        ).squeeze()

        box += torch.cat((box_pos_embed[y_coord, x_coord], box_pos_embed[h, w]), dim=2)
        box = self.sem_embed(box, bbox_info[:, :, 0], bbox=True)
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
        self.__image_size = (h, w)
        pw, ph = self.get_patch_size()
        gs_w = w // pw
        gs_h = h // ph

        posemb_tok, posemb_grid = (
            self.pos_embedding[:, :start_index],
            self.pos_embedding[0, start_index:],
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
            bbox_info (torch.Tensor): Bounding box information tensor of shape (batch_size, num_boxes, 5).
                Default is None.
            n_bbox (int): Number of bounding boxes. Default is -1.

        Returns:
            aggregated_feat (torch.Tensor): Aggregated feature tensor of shape (batch_size, num_patches, hidden_dim).
            aggregated_box (torch.Tensor): Aggregated box tensor of shape (batch_size, num_boxes, hidden_dim).
                None if n_bbox <= 0.
        """
        # Apply the patch embedding
        patch_embed_x = self.patch_embed(x).to(x.device)

        # Add semantic embeddings
        if sem_embed:
            if sem_labels is None:
                sem_labels = torch.zeros(
                    patch_embed_x.shape[0], patch_embed_x.shape[1], dtype=torch.long
                ).to(x.device)
            patch_embed_x = self.sem_embed(patch_embed_x, sem_labels)

        patch_embed_x = patch_embed_x + self.pos_embed.to(x.device)

        # Add box embeddings
        if bbox_info is not None:
            box_feat = self.extract_box_feature(x, bbox_info, n_bbox, self.patch_size)
            patch_embed_x = self.add_box(patch_embed_x, box_feat, bbox_info, n_bbox)

        return patch_embed_x
        # Concatenate class tokens and embedded features
        # Add positional embeddings except for the box embeddings


def posemb_sincos_2d(patches, temperature=10000, dtype=torch.float32):
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.meshgrid(
        torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij"
    )
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4, device=device) / (dim // 4 - 1)
    omega = 1.0 / (temperature**omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


class PositionalEncoding(nn.Module):
    """Positional Encoding module injects information about the relative position of the tokens in the sequence.

    Args:
      input_dim:    dimension of embeddings
      dropout:      randomly zeroes-out some of the input
      max_length:   max sequence length
    """

    def __init__(self, input_dim: int, dropout: float = 0.1, max_len: int = 5000):
        # input_dim: dimension of the model
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        position = torch.arange(max_len).unsqueeze(1)
        # 1,5000
        div_term = torch.exp(
            torch.arange(0, input_dim, 2) * (-math.log(10000.0) / input_dim)
        )
        pos_emb = torch.zeros(max_len, 1, input_dim)
        # 5000,1,1024
        # calc sine on even indices
        pos_emb[:, 0, 0::2] = torch.sin(position * div_term)
        # 5000,1,1024
        # calc cosine on odd indices
        pos_emb[:, 0, 1::2] = torch.cos(position * div_term)

        pos_emb = pos_emb.permute(1, 0, 2)

        # registered buffers are saved in state_dict but not trained by the optimizer
        self.register_buffer("pos_emb", pos_emb)

    def forward(self) -> torch.Tensor:
        """
        Forward pass of PositionalEncoding module.
        """
        # self.pos_emb = self.pos_emb.permute(1, 0, 2)
        return self.dropout(self.pos_emb)


class PatchEmbedding(nn.Module):
    """2D Image to Patch Embedding.
    Patch embedding layer used in Vision Transformer (https://arxiv.org/abs/2010.11929)
    """

    def __init__(
        self,
        input_size=96,
        patch_size=8,
        in_chans=256,
        embed_dim=768,
        norm_layer=None,
        STEM=True,
    ):
        super().__init__()
        # image and patch size to tuple of 2 integers
        to_2tuple = _ntuple(2)
        input_size = to_2tuple(input_size)
        patch_size = to_2tuple(patch_size)
        self.input_size = input_size
        # logging.info(f"PatchEmbedding input_size: {input_size}, in_chans: {in_chans}, embed_dim: {embed_dim}")
        # self.patch_size = patch_size
        # grid size is the number of patches in the image
        # self.grid_size = (input_size[0] // patch_size[0], input_size[1] // patch_size[1])
        # self.num_patches = self.grid_size[0] * self.grid_size[1]

        # if STEM is true, we use a stem conv layer to project the input image to a feature map of size embed_dim
        # stem conv layer is a 3x3 conv layer with stride 2, followed by a 3x3 conv layer with stride 2, followed by a 3x3 conv layer with stride 2, followed by a 1x1 conv layer
        if STEM:
            hidden_dim = embed_dim // in_chans
            self.proj = nn.Sequential(
                nn.Conv2d(
                    in_chans,
                    in_chans * hidden_dim // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                nn.Conv2d(
                    in_chans * hidden_dim // 2,
                    in_chans * hidden_dim // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                nn.Conv2d(
                    in_chans * hidden_dim // 2,
                    embed_dim,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
            )
        else:  # we use standart ViT patch embedding layer
            self.proj = nn.Conv2d(
                in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
            )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _, H, W = x.shape
        assert (
            H == self.input_size[0] and W == self.input_size[1]
        ), f"Input image size ({H}*{W}) doesn't match expected model input ({self.input_size[0]}*{self.input_size[1]})"
        x = self.proj(x).flatten(2).transpose(1, 2)
        # proj:4,1024,8,8 #fl:4,1024,64
        # proj:120,1024,1,1  fl:120,1024,1 tr:120,1,1024
        x = self.norm(x)
        return x


class SemanticEmbedding(nn.Module):
    def __init__(self, num_sem_classes, embed_dim, patch_size=8, min_coverage=0.6, content_dim = 256):
        super().__init__()
        self.embedding = nn.Embedding(num_sem_classes, embed_dim)
        self.to_patches = transforms.Compose(
            [SegMaskToPatches(patch_size=patch_size, min_coverage=min_coverage, input_dim=content_dim)]
        )

    def forward(self, x, sem_labels, bbox=False):
        if not bbox:
            sem_labels = self.to_patches(sem_labels).to((x.device))
        semantic_emb = self.embedding(sem_labels.to(torch.long))
        assert (
            x.shape[1] == semantic_emb.shape[1]
        ), f"Channel dim of input {x.shape[1]} and semantic embeddings {semantic_emb.shape[1]} must match."
        x = torch.cat([x, semantic_emb], dim=2)
        return x
