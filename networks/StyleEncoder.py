import math
import numpy as np
import torch
import torch.nn as nn

from . import blocks

class StyleEncoder(nn.Module):
    def __init__(self, input_channels=3, generator_in_filters=64, style_dim=8, n_downsampling=2, no_antialias=False, num_domains=2):
        super(StyleEncoder, self).__init__()

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_channels, generator_in_filters, kernel_size=7, padding=0, bias=True),
                 nn.InstanceNorm2d(generator_in_filters),
                 nn.ReLU(True)]

        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            if(no_antialias):
                model += [nn.Conv2d(generator_in_filters * mult, generator_in_filters * mult * 2, kernel_size=3, stride=2, padding=1, bias=True),
                          nn.InstanceNorm2d(generator_in_filters * mult * 2),
                          nn.ReLU(True)]
            else:
                model += [nn.Conv2d(generator_in_filters * mult, generator_in_filters * mult * 2, kernel_size=3, stride=1, padding=1, bias=True),
                          nn.InstanceNorm2d(generator_in_filters * mult * 2),
                          nn.ReLU(True),
                          blocks.Downsample(generator_in_filters * mult * 2)]
        
        model += [nn.AdaptiveAvgPool2d(1)]
        model += [nn.Conv2d(generator_in_filters * mult * 2, style_dim, 1, 1, 0)]    
        self.model = nn.Sequential(*model)
        # Unshared layers for domain-specific transformation 
        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Linear(generator_in_filters * mult * 2, style_dim)]        
        
    def forward(self, x, y=None):
        h = self.model(x)
        # h = h.view(h.size(0), -1)
        # out = []
        # for layer in self.unshared:
            # out += [layer(h)]
        # out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        # idx = torch.LongTensor(range(y.size(0))).to(y.device)
        # s = out[idx, y]  # (batch, style_dim)
        return h

class StyleEncoder_v2(nn.Module):
    def __init__(self, img_size=256, style_dim=64, num_domains=2, max_conv_dim=512):
        super().__init__()
        dim_in = 2**14 // img_size
        blocks = []
        blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Linear(dim_out, style_dim)]

    def forward(self, x, y):
        h = self.shared(x)
        h = h.view(h.size(0), -1)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        s = out[idx, y]  # (batch, style_dim)
        return s



class MLP(nn.Module):
    def __init__(self, input_dim=8, output_dim=2048, dim=256, n_blk=3, norm='none', activ='relu'):

        super(MLP, self).__init__()
        self.model = []
        self.model += [blocks.LinearBlock(input_dim, dim, norm=norm, activation=activ)]
        for i in range(n_blk - 2):
            self.model += [blocks.LinearBlock(dim, dim, norm=norm, activation=activ)]
        self.model += [blocks.LinearBlock(dim, output_dim, norm='none', activation='none')] # no output activations
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))
    
#from StarGAN v2
class MappingNetwork(nn.Module):
# Similar to the one in StarGan
    def __init__(self, latent_dim=16, style_dim=64, num_domains=2):
        super().__init__()
        layers = []
        layers += [nn.Linear(latent_dim, 512)]
        layers += [nn.ReLU()]
        for _ in range(3):
            layers += [nn.Linear(512, 512)]
            layers += [nn.ReLU()]
        self.shared = nn.Sequential(*layers)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Sequential(nn.Linear(512, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, style_dim))]

    def forward(self, z, y):
        h = self.shared(z)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        s = out[idx, y]  # (batch, style_dim)
        return s    
        
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