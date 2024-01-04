import functools 
import torch
import torch.nn as nn

from . import blocks

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_channels=3, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, no_antialias=False):
        """Construct a PatchGAN discriminator

        Parameters:
            input_channels (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        if(no_antialias):
            sequence = [nn.Conv2d(input_channels, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        else:
            sequence = [nn.Conv2d(input_channels, ndf, kernel_size=kw, stride=1, padding=padw), nn.LeakyReLU(0.2, True), blocks.Downsample(ndf)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            if(no_antialias):
                sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True)
                ]
            else:
                sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True),
                    blocks.Downsample(ndf * nf_mult)]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)
    
    #StarGan Discriminator
    class Discriminator(nn.Module):
        def __init__(self, img_size=256, num_domains=2, max_conv_dim=512):
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
            blocks += [nn.Conv2d(dim_out, num_domains, 1, 1, 0)]
            self.main = nn.Sequential(*blocks)

        def forward(self, x, y):
            out = self.main(x)
            out = out.view(out.size(0), -1)  # (batch, num_domains)
            idx = torch.LongTensor(range(y.size(0))).to(y.device)
            out = out[idx, y]  # (batch)
            return out