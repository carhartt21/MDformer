import torch
import torch.nn as nn

from . import blocks

class StyleEncoder(nn.Module):
    def __init__(self, input_nc=3, ngf=64, style_dim=8, n_downsampling=2, no_antialias=False, num_domains=2):
        super(StyleEncoder, self).__init__()

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=True),
                 nn.InstanceNorm2d(ngf),
                 nn.ReLU(True)]

        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            if(no_antialias):
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=True),
                          nn.InstanceNorm2d(ngf * mult * 2),
                          nn.ReLU(True)]
            else:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=True),
                          nn.InstanceNorm2d(ngf * mult * 2),
                          nn.ReLU(True),
                          blocks.Downsample(ngf * mult * 2)]
        
        model += [nn.AdaptiveAvgPool2d(1)]
        model += [nn.Conv2d(ngf * mult * 2, style_dim, 1, 1, 0)]    
        self.model = nn.Sequential(*model)
        # Unshared layers for domain-specific transformation 
        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Linear(ngf * mult * 2, style_dim)]        
        
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