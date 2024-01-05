import math
from networkx import spectral_ordering

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import numpy as np

####################################    Residual Block   ####################################
class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample=False):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance

####################################    Adaptive Instance Normalization  ####################################
class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class AdainResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64, w_hpf=0,
                 actv=nn.LeakyReLU(0.2), upsample=False):
        super().__init__()
        self.w_hpf = w_hpf
        self.actv = actv
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)

    def _build_weights(self, dim_in, dim_out, style_dim=64):
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        if self.w_hpf == 0:
            out = (out + self._shortcut(x)) / math.sqrt(2)
        return out


class HighPass(nn.Module):
    def __init__(self, w_hpf, device):
        super(HighPass, self).__init__()
        self.register_buffer('filter',
                             torch.tensor([[-1, -1, -1],
                                           [-1, 8., -1],
                                           [-1, -1, -1]]) / w_hpf)

    def forward(self, x):
        filter = self.filter.unsqueeze(0).unsqueeze(1).repeat(x.size(1), 1, 1, 1)
        return F.conv2d(x, filter, padding=1, groups=x.size(1))



####################################    Transformer   ####################################

def posemb_sincos_2d(patches, temperature = 10000, dtype = torch.float32):
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.meshgrid(torch.arange(h, device = device), torch.arange(w, device = device), indexing = 'ij')
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(dim // 4, device = device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :] 
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = 1)
    return pe.type(dtype)

class PositionalEncoding(nn.Module):
    """ Positional Encoding module injects information about the relative position of the tokens in the sequence.

    Args:
      d_model:      dimension of embeddings
      dropout:      randomly zeroes-out some of the input
      max_length:   max sequence length
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        # d_model: dimension of the model
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        position = torch.arange(max_len).unsqueeze(1)
        # 1,5000
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))        
        pos_emb = torch.zeros(max_len, 1, d_model)
        # 5000,1,1024
        # calc sine on even indices
        pos_emb[:, 0, 0::2] = torch.sin(position * div_term)
        # 5000,1,1024
        # calc cosine on odd indices       
        pos_emb[:, 0, 1::2] = torch.cos(position * div_term)
        # registered buffers are saved in state_dict but not trained by the optimizer           
        self.register_buffer('pos_emb', pos_emb)

    def forward(self) -> Tensor:
        """ 
        Forward pass of PositionalEncoding module.
        """
        self.pos_emb = self.pos_emb.permute(1, 0, 2)
        return self.dropout(self.pos_emb)


####################################    Norm   ####################################

class AdaptiveInstanceNorm2d(nn.Module):
    """ AdaptiveInstanceNorm2d module implements adaptive instance normalization layer.
    Args:
        num_features: num_features default 1024
        eps: epsilon default 1e-5
        momentum: momentum default 0.1
    """
    def __init__(self, num_features: int, eps: float = 1e-5, momentum:float = 0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        # num_features: 1024
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x): #x: 4,79,1024
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        
        x = x.transpose(1,2)
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b) #4096 #tensor([0., 0., 0.,  ..., 0., 0., 0.], device='cuda:0')  
        running_var = self.running_var.repeat(b) #4096 #tensor([1., 1., 1.,  ..., 1., 1., 1.], device='cuda:0')

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:]) #([1, 316, 1024]) 
        #1,2048,84
        out = F.batch_norm(x_reshaped, running_mean, running_var, self.weight, self.bias,True, self.momentum, self.eps)
        # out: 1,4096,79
        
        return out.view(b, c, *x.size()[2:]).transpose(1,2)

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


class LayerNorm(nn.Module):
    """ Layer normalization."""
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x
####################################    Norm   ####################################

####################################    MGUIT   ####################################

class ReLUINSConvTranspose2d(nn.Module):
    """ ReLU + InstanceNorm + ConvTranspose2d."""
    def __init__(self, n_in, n_out, kernel_size, stride, padding, output_padding):
        super(ReLUINSConvTranspose2d, self).__init__()
        model = []
        model += [nn.ConvTranspose2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=True)]
        # model += [nn.InstanceNorm2d(n_out, affine=False)]
        model += [LayerNorm(n_out)]
        model += [nn.ReLU(inplace=True)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)
    def forward(self, x):
        return self.model(x)

class LeakyReLUConv2d(nn.Module):
    """ LeakyReLU + Conv2d."""
    def __init__(self, n_in, n_out, kernel_size, stride, padding=0, norm='None', sn=False):
        super(LeakyReLUConv2d, self).__init__()
        model = []
        model += [nn.ReflectionPad2d(padding)]
        if sn:
            # spectral_norm(nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=0, bias=True))
            model += [spectral_ordering(nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=0, bias=True))]
        else:
            model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=0, bias=True)]
        if 'norm' == 'Instance':
            model += [nn.InstanceNorm2d(n_out, affine=False)]
        model += [nn.LeakyReLU(inplace=True)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)
        #elif == 'Group'
    def forward(self, x):
        return self.model(x)

class ReLUINSConv2d(nn.Module):
    """ ReLU + InstanceNorm + Conv2d +ReflectionPad2d """
    def __init__(self, n_in, n_out, kernel_size, stride, padding=0):
        super(ReLUINSConv2d, self).__init__()
        model = []
        model += [nn.ReflectionPad2d(padding)]
        model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=0, bias=True)]
        model += [nn.InstanceNorm2d(n_out, affine=False)]
        model += [nn.ReLU(inplace=True)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)
    def forward(self, x):
        return self.model(x)

def gaussian_weights_init(m):
    """ Initialize conv weights with Gaussian distribution."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Conv') == 0:
        m.weight.data.normal_(0.0, 0.02)

####################################    MGUIT   ####################################
####################################    Style   ####################################

def get_filter(filt_size=3):
    """ Get filter of size filt_size."""
    if(filt_size == 1):
        a = np.array([1., ])
    elif(filt_size == 2):
        a = np.array([1., 1.])
    elif(filt_size == 3):
        a = np.array([1., 2., 1.])
    elif(filt_size == 4):
        a = np.array([1., 3., 3., 1.])
    elif(filt_size == 5):
        a = np.array([1., 4., 6., 4., 1.])
    elif(filt_size == 6):
        a = np.array([1., 5., 10., 10., 5., 1.])
    elif(filt_size == 7):
        a = np.array([1., 6., 15., 20., 15., 6., 1.])

    filt = torch.Tensor(a[:, None] * a[None, :])
    filt = filt / torch.sum(filt)

    return filt

def get_pad_layer(pad_type):
    """ Get padding layer."""
    if(pad_type in ['refl', 'reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl', 'replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type == 'zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized' % pad_type)
    return PadLayer

class Downsample(nn.Module):
    """ Downsample module."""
    def __init__(self, channels, pad_type='reflect', filt_size=3, stride=2, pad_off=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2)), int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size == 1):
            if(self.pad_off == 0):
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])



class LinearBlock(nn.Module):
    """ Linear block module. Functions: fully connected + normalization + activation."""
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'batch':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'inst':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out


####################################    Style   ####################################


####################################    Aggregator   ####################################

from itertools import repeat
import collections.abc

# Convert a number to a tuple of n
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

# Convert a number to a tuple of 2
# Example: 1 -> (1, 1)
to_2tuple = _ntuple(2)

class PatchEmbedding(nn.Module):
    """ 2D Image to Patch Embedding. 
    Patch embedding layer used in Vision Transformer (https://arxiv.org/abs/2010.11929)
    """

    def __init__(self, input_size=64, patch_size=8, in_chans=256, embed_dim=768, norm_layer=None, STEM=True):
        super().__init__()
        # image and patch size to tuple of 2 integers
        input_size = to_2tuple(input_size)
        patch_size = to_2tuple(patch_size)
        self.input_size = input_size
        self.patch_size = patch_size
        # grid size is the number of patches in the image
        self.grid_size = (input_size[0] // patch_size[0], input_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        # if STEM is true, we use a stem conv layer to project the input image to a feature map of size embed_dim
        # stem conv layer is a 3x3 conv layer with stride 2, followed by a 3x3 conv layer with stride 2, followed by a 3x3 conv layer with stride 2, followed by a 1x1 conv layer
        if STEM:
            hidden_dim = embed_dim // in_chans
            self.proj = nn.Sequential(
                nn.Conv2d(in_chans, in_chans*hidden_dim//2, kernel_size=3, stride=2, padding=1),
                nn.Conv2d(in_chans*hidden_dim//2, in_chans*hidden_dim//2, kernel_size=3, stride=2, padding=1),
                nn.Conv2d(in_chans*hidden_dim//2, embed_dim, kernel_size=3, stride=2, padding=1),
                nn.Conv2d(embed_dim, embed_dim, kernel_size=1))
        else: # we use standart ViT patch embedding layer
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size) 
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x): #x: b*2,256,64,64 / b*2*30,256,8,8
        _, _, H, W = x.shape
        assert H == self.input_size[0] and W == self.input_size[1], \
            "Input image size ({}*{}) doesn't match model ({}*{})".format(H, W, self.input_size[0], self.input_size[1])
        x = self.proj(x).flatten(2).transpose(1, 2) #proj:4,1024,8,8 #fl:4,1024,64
        #proj:120,1024,1,1  fl:120,1024,1 tr:120,1,1024
        x = self.norm(x)
        return x
    
class SemanticEmbedding(nn.Module):
    def __init__(self, in_dim: int=64, input_size: int = 88, patch_size: int = 8, cls_labels: list=[], out_dim: int=256):
        super().__init__()
        input_size = to_2tuple(input_size)
        patch_size = to_2tuple(patch_size)
        self.input_size = input_size
        self.patch_size = patch_size
        self.fc = nn.Linear(in_dim, out_dim)
        self.norm = LayerNorm(out_dim)
        self.grid_size = (input_size[0] // patch_size[0], input_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]        
    
    def forward(self, x):
        x = self.fc(x)
        x = self.norm(x)
        return x

    def create_embedded_token(self, x, c):
        # Create an embedded token based on input tensor x and semantic class c
        embedded_token = torch.cat((x, c), dim=1)
        return embedded_token

####################################    Aggregator   ####################################


####################################    Generator   ####################################

class Transpose(nn.Module):
    # Transpose layer; dim0 and dim1 are the dimension to be swapped
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        x = x.transpose(self.dim0, self.dim1)
        return x
    
class Upsample(nn.Module):
    # Upsample the image by a factor of stride, channels is the number of channels in the input image
    def __init__(self, channels, pad_type='repl', filt_size=4, stride=2):
        super(Upsample, self).__init__()
        self.filt_size = filt_size
        self.filt_odd = np.mod(filt_size, 2) == 1
        self.pad_size = int((filt_size - 1) / 2)
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size) * (stride**2)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        self.pad = get_pad_layer(pad_type)([1, 1, 1, 1])

    def forward(self, inp):
        ret_val = F.conv_transpose2d(self.pad(inp), self.filt, stride=self.stride, padding=1 + self.pad_size, groups=inp.shape[1])[:, :, 1:, 1:]
        if(self.filt_odd):
            return ret_val
        else:
            return ret_val[:, :, :-1, :-1]

####################################    Generator   ####################################


####################################    Discriminator   ####################################

class Downsample(nn.Module):
    # Downsample the image by a factor of stride, channels is the number of channels in the input image
    def __init__(self, channels, pad_type='reflect', filt_size=3, stride=2, pad_off=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2)), int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size == 1):
            if(self.pad_off == 0):
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])

####################################    Discriminator   ####################################


####################################    MLP Head   ####################################

class Normalize(nn.Module):
    # Normalization layer from the StyleGAN2 discriminator
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out

####################################    MLP Head   ####################################