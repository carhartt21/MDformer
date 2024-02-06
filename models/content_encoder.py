import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from . import blocks


class ContentEncoder(nn.Module):
    def __init__(
        self,
        input_channels=3,
        n_generator_filters=64,
        n_downsampling=2,
        no_antialias=False,
    ):
        """
        n_generator_filters = 64 number of generator filters
        input_channels = 3 number of input channels
        n_downsampling = 2 number of downsampling layers
        no_antialias = False use antialiasing in downsampling layers
        """

        super(ContentEncoder, self).__init__()

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(
                input_channels, n_generator_filters, kernel_size=7, padding=0, bias=True
            ),
            nn.InstanceNorm2d(n_generator_filters),
            nn.ReLU(True),
        ]

        for i in range(n_downsampling):  # add downsampling layers
            mult = 2**i
            if no_antialias:
                model += [
                    nn.Conv2d(
                        n_generator_filters * mult,
                        n_generator_filters * mult * 2,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=True,
                    ),
                    nn.InstanceNorm2d(n_generator_filters * mult * 2),
                    nn.ReLU(True),
                ]
            else:
                model += [
                    nn.Conv2d(
                        n_generator_filters * mult,
                        n_generator_filters * mult * 2,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=True,
                    ),
                    nn.InstanceNorm2d(n_generator_filters * mult * 2),
                    nn.ReLU(True),
                    blocks.Downsample(n_generator_filters * mult * 2),
                ]

        self.model = nn.Sequential(*model)

    def forward(self, x, layers=[]):
        features = []
        feature = x

        for i, layer in enumerate(self.model):
            feature = layer(feature)

            if i in layers:
                features.append(feature)
        out = feature
        return out, features


class ContentEncoder2(nn.Module):
    def __init__(
        self,
        input_channels=3,        
        n_downsampling=2,
        num_res_blocks=3,
        input_dim=512,
        ngf=128,
        norm="IN",
        activation="ReLU",
        pad_type="reflect",
        no_antialias=False,
    ):
        super(ContentEncoder2, self).__init__()
        logging.info(f"ContentEncoder2 input_channels: {input_channels}, n_downsampling: {n_downsampling}, num_res_blocks: {num_res_blocks}, input_dim: {input_dim}, ngf: {ngf}, norm: {norm}, activation: {activation}, pad_type: {pad_type}, no_antialias: {no_antialias}")
        model = []
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(
                input_channels, ngf, kernel_size=7, padding=0, bias=True
            ),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
        ]
        # downsampling blocks
        for i in range(n_downsampling):
            mult = 2**i
            if no_antialias:
                model += [
                    nn.Conv2d(
                        ngf * mult,
                        ngf * mult * 2,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=True,
                    ),
                    nn.InstanceNorm2d(ngf * mult * 2),
                    nn.ReLU(True),
                ]
            else:
                model += [
                    nn.Conv2d(
                        ngf * mult,
                        ngf * mult * 2,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=True,
                    ),
                    nn.InstanceNorm2d(ngf * mult * 2),
                    nn.ReLU(True),
                    blocks.Downsample(ngf * mult * 2),
                ]

        mult = 2 ** n_downsampling

        # residual blocks
        for i in range(num_res_blocks):
            model += [
                ResBlock(ngf * mult, norm=norm, activation=activation, pad_type=pad_type)
            ]
        self.model = nn.Sequential(*model)
        # self.ngf = dim

    def forward(self, x, layers=[]):
        features = []
        feature = x
        for i, layer in enumerate(self.model):
            feature = layer(feature)
            if i in layers:
                features.append(feature)
        out = feature
        logging.info(f"ContentEncoder2 output shape: {out.shape}")
        return out, features


##################################################################################
# Basic Blocks
##################################################################################
class ResBlocks(nn.Module):
    def __init__(
        self, num_res_blocks, dim, norm="in", activation="relu", pad_type="zero"
    ):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_res_blocks):
            self.model += [
                ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type)
            ]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class ResBlock(nn.Module):
    def __init__(self, dim, norm="in", activation="relu", pad_type="zero"):
        super(ResBlock, self).__init__()

        model = []
        model += [
            Conv2dBlock(
                dim, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type
            )
        ]
        model += [
            Conv2dBlock(
                dim, dim, 3, 1, 1, norm=norm, activation="none", pad_type=pad_type
            )
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = x + self.model(x)
        return out


class Conv2dBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        kernel_size,
        stride,
        padding=0,
        norm="none",
        activation="relu",
        pad_type="zero",
    ):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == "reflect":
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == "replicate":
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == "zero":
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == "BN":
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == "IN":
            # self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == "AdaIN":
            self.norm = blocks.AdaptiveInstanceNorm2d(norm_dim)
        elif norm == "None" or norm == "SN":
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == "ReLU":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "LeakyReLU":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "PReLU":
            self.activation = nn.PReLU()
        elif activation == "GeLU":
            self.activation = nn.GeLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "none":
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        self.conv = nn.Conv2d(
            input_dim, output_dim, kernel_size, stride, bias=self.use_bias
        )

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


# ContentEncoder(
# (model): Sequential(
#     (0): ReflectionPad2d((3, 3, 3, 3))
#     (1): Conv2d(3, 64, kernel_size=(7, 7), stride=(1, 1))
#     (2): InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
#     (3): ReLU(inplace=True)
#     (4): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (5): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
#     (6): ReLU(inplace=True)
#     (7): Downsample(
#     (pad): ReflectionPad2d([1, 1, 1, 1])
#     )
#     (8): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (9): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
#     (10): ReLU(inplace=True)
#     (11): Downsample(
#     (pad): ReflectionPad2d([1, 1, 1, 1])
#     )
# )
# )


# MGUIT
# class ContentEncoder(nn.Module):
#     def __init__(self, input_dim_a=3, tch=64):
#         super(ContentEncoder, self).__init__()
#         encA_c = []

#         encA_c += [blocks.LeakyReLUConv2d(input_dim_a, tch, kernel_size=7, stride=1, padding=3)]
#         for i in range(1, 3):
#             encA_c += [blocks.ReLUINSConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1)]
#             tch *= 2

#         self.convA = nn.Sequential(*encA_c)

#     def forward(self, xa): #xa: 8,3,256,256
#         outputA = self.convA(xa)
#         return F.normalize(outputA, dim=1)

# (convA): Sequential(
#       (0): LeakyReLUConv2d(
#         (model): Sequential(
#           (0): ReflectionPad2d((3, 3, 3, 3))
#           (1): Conv2d(3, 64, kernel_size=(7, 7), stride=(1, 1))
#           (2): LeakyReLU(negative_slope=0.01, inplace=True)
#         )
#       )
#       (1): ReLUINSConv2d(
#         (model): Sequential(
#           (0): ReflectionPad2d((1, 1, 1, 1))
#           (1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2))
#           (2): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
#           (3): ReLU(inplace=True)
#         )
#       )
#       (2): ReLUINSConv2d(
#         (model): Sequential(
#           (0): ReflectionPad2d((1, 1, 1, 1))
#           (1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2))
#           (2): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
#           (3): ReLU(inplace=True)
#         )
#       )
#     )
