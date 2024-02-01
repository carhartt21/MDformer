import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import logging
from . import blocks


def init_weights(net, init_type="normal", init_gain=0.02, debug=False):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if debug:
                print(classname)
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif (
            classname.find("BatchNorm2d") != -1
        ):  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(
    net,
    init_type="normal",
    init_gain=0.02,
    gpu_ids=[],
    debug=False,
    initialize_weights=True,
):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert torch.cuda.is_available()
        net.to(gpu_ids[0])
        # if not amp:# modified
        # net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs for non-AMP training
    if initialize_weights:
        init_weights(net, init_type, init_gain=init_gain, debug=debug)
    return net


class MLPHead(nn.Module):
    """
    Multi-Layer Perceptron (MLP) head module.

    Args:
        use_mlp (bool): Whether to use the MLP head or not. Default is True.
        init_type (str): Type of weight initialization. Default is 'xavier'.
        init_gain (float): Gain factor for weight initialization. Default is 0.02.
        nc (int): Number of output channels. Default is 256.

    Attributes:
        l2norm (blocks.Normalize): L2 normalization layer.
        use_mlp (bool): Whether to use the MLP head or not.
        nc (int): Number of output channels.
        mlp_init (bool): Whether the MLP has been initialized or not.
        init_type (str): Type of weight initialization.
        init_gain (float): Gain factor for weight initialization.

    Methods:
        create_mlp(feats, device): Creates the MLP layers.
        forward(feats, num_patches=64, patch_ids=None, contrastive_mlp=True): Forward pass of the MLP head.

    """

    # potential TODO: add option to use MLP head for classification
    # TODO: use different patch_ids for different min batches

    def __init__(self, use_mlp=True, init_type="xavier", init_gain=0.02, nc=256):
        super(MLPHead, self).__init__()
        self.l2norm = blocks.Normalize(2)
        self.use_mlp = use_mlp
        self.nc = nc
        self.mlp_init = False
        self.init_type = init_type
        self.init_gain = init_gain

    def create_mlp(self, feats, device):
        """
        Creates the MLP layers.

        Args:
            feats (list): List of input feature maps.
            device (torch.device): Device to place the MLP layers on.

        """
        for mlp_id, feat in enumerate(feats):
            try:
                if len(feat.shape) == 4:  # Conv
                    input_channels = feat.shape[1]
                else:  # Attention
                    input_channels = feat.shape[2]
            except:
                feat = feat[0]
                if len(feat.shape) == 4:  # Conv
                    input_channels = feat.shape[1]
                else:  # Attention
                    input_channels = feat.shape[2]

            mlp = nn.Sequential(
                *[
                    nn.Linear(input_channels, self.nc),
                    nn.ReLU(),
                    nn.Linear(self.nc, self.nc),
                ]
            )

            mlp.to(device)
            setattr(self, "mlp_%d" % mlp_id, mlp)
        init_net(self, self.init_type, self.init_gain, [0])  # 0 -> device
        self.mlp_init = True

    def forward(
        self,
        feats,
        num_patches=64,
        patch_ids=None,
        contrastive_mlp=True,
        sem_feats=None,
    ):
        """
        Forward pass of the MLP head.

        Args:
            feats (list): List of input feature maps.
            num_patches (int): Number of patches to sample. Default is 64.
            patch_ids (list or None): List of patch IDs to sample from each feature map. Default is None.
            contrastive_mlp (bool): Whether to apply the MLP layers for contrastive learning. Default is True.

        Returns:
            return_feats (list): List of output feature maps.
            return_ids (list): List of patch IDs.

        """
        return_ids = []
        return_feats = []
        # batch_size = feats[0].shape[0]
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feats, feats[0].device)
        for feat_id, feat in enumerate(feats):
            if len(feat.shape) == 4:  # Conv
                B, H, W = (
                    feat.shape[0],
                    feat.shape[2],
                    feat.shape[3],
                )  # torch.Size([2, 128, 256, 256])    #inst 2,256,8,8
                feat_reshape = feat.permute(0, 2, 3, 1).flatten(
                    1, 2
                )  # torch.Size([2, 65536, 128]) ##inst [2, 64, 256]
            else:  # Attention
                feat_reshape = feat
            if num_patches > 0:
                if patch_ids is not None:
                    patch_id = patch_ids[feat_id]
                else:
                    patch_id = torch.randperm(
                        feat_reshape.shape[1], device=feats[0].device
                    )
                    patch_id = patch_id[
                        : int(min(num_patches, patch_id.shape[0]))
                    ]  # .to(patch_ids.device)
                x_sample = feat_reshape[:, patch_id, :].flatten(
                    0, 1
                )
            else:
                x_sample = feat_reshape  # inst 128,256
                patch_id = []

            if self.use_mlp and contrastive_mlp:
                mlp = getattr(self, "mlp_%d" % feat_id)

                x_sample = x_sample.cuda().to(feats[0].device)
                x_sample = mlp(x_sample)

            return_ids.append(patch_id)
            x_sample = self.l2norm(x_sample)

            if num_patches == 0:
                x_sample = x_sample.permute(0, 2, 1).reshape(
                    [B, x_sample.shape[-1], H, W]
                )
            return_feats.append(x_sample)
        return return_feats, return_ids


class NPMLPHead(nn.Module):
    def __init__(
        self, use_mlp=True, init_type="xavier", init_gain=0.02, nc=256, gpu_ids=[]
    ):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super(NPMLPHead, self).__init__()
        self.l2norm = blocks.Normalize(2)
        self.use_mlp = use_mlp
        self.nc = nc  # hard-coded
        self.mlp_init = False
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids

    def create_mlp(self, feats, device):
        """
        Creates the MLP layers.

        Args:
            feats (list): List of input feature maps.
            device (torch.device): Device to place the MLP layers on.

        """
        for mlp_id, feat in enumerate(feats):
            try:
                if len(feat.shape) == 4:  # Conv
                    input_channels = feat.shape[1]
                else:  # Attention
                    input_channels = feat.shape[2]
            except:
                feat = feat[0]
                if len(feat.shape) == 4:  # Conv
                    input_channels = feat.shape[1]
                else:  # Attention
                    input_channels = feat.shape[2]

            mlp = nn.Sequential(
                *[
                    nn.Linear(input_channels, self.nc),
                    nn.ReLU(),
                    nn.Linear(self.nc, self.nc),
                ]
            )

            mlp.to(device)
            setattr(self, "mlp_%d" % mlp_id, mlp)
        init_net(self, self.init_type, self.init_gain, [0])  # 0 -> device
        self.mlp_init = True

    def forward(self, feats, seg=None, num_patches = 128, num_pos=32, num_neg=64, patch_ids=None):
        num_patches = 128
        return_ids = []
        return_feats = []
        pos_feats = []
        neg_feats = []
        assert seg is not None if patch_ids is not None else True
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feats, feats[0].device)
        for feat_id, feat in enumerate(feats):
            B, H, W = feat.shape[0], feat.shape[2], feat.shape[3]
            feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2).squeeze(0)
            if seg is not None:
                seg_reshape = seg.type(torch.cuda.FloatTensor)
                seg_reshape = F.interpolate(seg_reshape, (H, W), mode="nearest"
                    ).squeeze(1)
                seg_reshape = seg_reshape.permute(0, 2, 1).flatten(1, 2)
            if num_patches > 0:
                layer_samples = []
                layer_pos_samples = []
                layer_neg_samples = []              
                if patch_ids is not None and seg is not None:
                    for b in range(B):
                        batch_pos_idx = []
                        batch_neg_idx = []                 
                        patch_id = patch_ids[feat_id]
                        for p_id in patch_id:
                            cls = seg_reshape[b, p_id]
                            mask = (seg_reshape[b] == cls) | (seg_reshape[b] == 0)
                            pos_features = feat_reshape[b, mask]
                            neg_features = feat_reshape[b, ~mask]
                            pos_id = torch.randperm(
                                pos_features.shape[0], device=feats[0].device
                            )
                            batch_pos_idx.append(pos_id[: int(min(num_pos, pos_id.shape[0]))])

                            neg_id = torch.randperm(
                                neg_features.shape[0], device=feats[0].device
                            )
                            batch_neg_idx.append(neg_id[: int(min(num_neg, neg_id.shape[0]))])
                        _pos_idx = torch.cat(batch_pos_idx, dim=0)
                        _neg_idx = torch.cat(batch_neg_idx, dim=0)
                        layer_samples.append(feat_reshape[b, patch_id, :])
                        layer_pos_samples.append(feat_reshape[b, _pos_idx, :])
                        layer_neg_samples.append(feat_reshape[b, _neg_idx, :]   )                        
                else:
                    patch_idx = torch.randperm(
                        feat_reshape.shape[1], device=feats[0].device
                    )
                    patch_idx = patch_idx[: int(min(num_patches, patch_idx.shape[0]))]
                    layer_samples = (feat_reshape[:, patch_idx, :].to(feats[0].device))
                    # reshape(-1, x.shape[1])
                    return_ids.append(patch_idx)
            else:
                layer_samples = feat_reshape
                return_ids = []
            if patch_ids is not None:
                layer_samples = torch.stack(layer_samples, dim=0)
            if layer_pos_samples != []:
                layer_pos_samples = torch.stack(layer_pos_samples, dim=0)
            if layer_neg_samples != []:
                layer_neg_samples = torch.stack(layer_neg_samples, dim=0)

            if self.use_mlp:
                mlp = getattr(self, "mlp_%d" % feat_id)
                layer_samples = mlp(layer_samples)
                layer_samples = self.l2norm(layer_samples)            
                if type(layer_pos_samples) == torch.Tensor:
                    layer_pos_samples = mlp(layer_pos_samples)
                    layer_neg_samples = self.l2norm(layer_neg_samples)

                if type(layer_neg_samples) == torch.Tensor:
                    layer_neg_samples = mlp(layer_neg_samples)
                    layer_pos_samples = self.l2norm(layer_pos_samples)

          
            if num_patches == 0:
                layer_samples = layer_samples.permute(0, 2, 1).reshape(
                    [B, layer_samples.shape[-1], H, W]
                )
            return_feats.append(layer_samples)
            pos_feats.append(layer_pos_samples)
            neg_feats.append(layer_neg_samples)
        return return_feats, return_ids, pos_feats, neg_feats 
