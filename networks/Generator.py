import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from . import blocks

class Generator(nn.Module):
    def __init__(self, input_size=96, patch_size=8, embed_C=1024, feat_C=256, n_generator_filters=64, n_downsampling=2, use_bias=True):
        super(Generator, self).__init__()
        

        self.inv_patch_embed = nn.Sequential(blocks.Transpose(1, 2),
                                    nn.Unflatten(2, torch.Size([input_size // patch_size, input_size // patch_size])),
                                    nn.ConvTranspose2d(in_channels=embed_C, out_channels=feat_C,
                                        kernel_size=patch_size, stride=patch_size, padding=0,
                                        bias=True, dilation=1, groups=1))
        self.inv_patch_embed_box = nn.Sequential(
                                    nn.ConvTranspose2d(in_channels=embed_C, out_channels=feat_C,
                                    kernel_size=patch_size, stride=patch_size, padding=0,
                                    bias=True, dilation=1, groups=1)
                                    )

        model = []
        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [blocks.Upsample(n_generator_filters * mult),
                    nn.Conv2d(n_generator_filters * mult, int(n_generator_filters * mult / 2),
                                kernel_size=3, stride=1,
                                padding=1,  # output_padding=1,
                                bias=use_bias),
                    nn.InstanceNorm2d(int(n_generator_filters * mult / 2)),
                    nn.ReLU(True)]

        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(n_generator_filters, 3, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)
        
    def forward(self, x, box_info=None):
        # remove the first token [<cls>]
        x = x[:, 1:, :]
        if box_info==None:
            out = self.model(self.inv_patch_embed(x))
        else:
            B_i, N_i, C_i = x.shape

            out_box_list = []
            box_index_list = []

            for i in range(B_i):
                out_box_filtered = x[i][torch.where(box_info[i,:,0] != 0)]
                box_index = out_box_filtered.shape[0]
                box_index_list.append(box_index) 
                if not(box_index == 0):            
                    out_box_list.append(out_box_filtered) 
            
            out_box = x.reshape(B_i*N_i,-1).unsqueeze(2).unsqueeze(3)
            out = self.model(self.inv_patch_embed_box(out_box))

        return out
