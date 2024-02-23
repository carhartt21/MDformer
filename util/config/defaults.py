from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
_C.DIR = "./weights"

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
# dataset name
_C.DATASET = CN()
# dataset mode (e.g., single, aligned, unaligned, multi)
_C.DATASET.mode = "multi"
# path to images (should have subfolders images, seg_labels, domain_labels)
_C.DATASET.root_dataset = "./data/"
# txt file containing training image folder
_C.DATASET.train_dir = "./data/train_images"
# path to the reference images
_C.DATASET.ref_dir = "./data/ref_images/"
# List of target domains
_C.DATASET.target_domain_names = ["summer", "winter", "spring", "autumn"]
# number of semantic classes
_C.DATASET.num_seg_class = 16
# number of domains
_C.DATASET.num_domains = 8
# maximum number of images per folder
_C.DATASET.max_dataset_size = 200000
# Maximum number of object boxes per image
# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# name of the model
_C.MODEL.name = "MDformer"
# transformer type ('vit'| 'swin')
_C.MODEL.transformer_type = "swin"
# load pretrained weights, False if training from scratch
_C.MODEL.load_weight = False
# load optimizer, False if training from scratch
_C.MODEL.load_optimizer = False
# path of pretrained weights. If training from scratch, set this as empty string
_C.MODEL.weight_path = ""
# weights to finetune net_decoder
# _C.MODEL.weights_decoder = ''
# number of input channels
_C.MODEL.in_channels = 3
# number of feature channels between encoder and decoder
_C.MODEL.fc_dim = 2048
# feature layers for NCE loss
_C.MODEL.feat_layers = [0, 4, 8]
# number of patches for NCE loss
_C.MODEL.num_patches = 256
# patch size
_C.MODEL.patch_size = 8
# number of negative samples for NCE loss
_C.MODEL.num_neg = 64
# number of positive samples for NCE loss
_C.MODEL.num_pos = 1
# input image size
_C.MODEL.img_size = (512, 512)
# Style code dimension
_C.MODEL.style_dim = 64
# Latent code dimension
_C.MODEL.latent_dim = 16
# dimension of the semantic embedding
_C.MODEL.sem_embed_dim = 32
# Mapping Network hidden dimension
_C.MODEL.hidden_dim = 512
# number of content channels
_C.MODEL.content_dim = 480
# number of downsampling layers in the content encoder
_C.MODEL.n_downsampling = 2
# number of input filters of the generator
_C.MODEL.n_generator_filters = 64
# number of discriminator filters
_C.MODEL.n_discriminator_filters = 64
# patch embed dimension
_C.MODEL.patch_embed_dim = 480
# ------------------------------------------------------------------------------
# Transformer parameters
# ------------------------------------------------------------------------------
_C.MODEL.VIT = CN()
# number of input channels of the transformer
_C.MODEL.VIT.feat_C = 256
# number of transformer layers
_C.MODEL.VIT.depth = 6
# number of heads in each transformer layer
_C.MODEL.VIT.heads = 4
# dimension of each head
_C.MODEL.VIT.mlp_dim = 4096
# dropout rate
_C.MODEL.VIT.dropout = 0.1
# ------------------------------------------------------------------------------
# SWIN Transformer parameters
# ------------------------------------------------------------------------------
_C.MODEL.SWIN = CN()
# patch size
_C.MODEL.SWIN.patch_size = 4
# embedding dimension
_C.MODEL.SWIN.embed_C = 128
# depth
_C.MODEL.SWIN.depths = [2, 2, 2]
# number of heads
_C.MODEL.SWIN.num_heads = [4, 8, 8]
# window sizes
_C.MODEL.SWIN.window_size = 8
# mlp ratio
_C.MODEL.SWIN.mlp_ratio = 4.0
# qkv bias
_C.MODEL.SWIN.qkv_bias = True
# absolute position embedding
_C.MODEL.SWIN.ape = False
# use absolute position embedding in swin transformer
_C.MODEL.SWIN.patch_norm = True
# drop path rate
_C.MODEL.SWIN.drop_path_rate = 0.1
# attention dropout rate
_C.MODEL.SWIN.attn_drop_rate = 0.0
# select in which stages patch mergin shall be enabled
_C.MODEL.SWIN.downsample = [True, True, False]


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
# pretrain the classifier
_C.TRAIN.pretrain = False
# seed to sample training images
_C.TRAIN.seed_train = 304
# distributed training
_C.TRAIN.distributed = True
# number of GPUs to use
_C.TRAIN.num_gpus = 1
# batch size per gpu
_C.TRAIN.batch_size_per_gpu = 4
# maximum number of epochs to train for
_C.TRAIN.num_epoch = 20
# number of workers to use for data loading
_C.TRAIN.num_workers = 6
# epoch to start training. useful if continue from a checkpoint
_C.TRAIN.start_epoch = 0
# epoch to end training
_C.TRAIN.end_epoch = 300
# iterations of each epoch (irrelevant to batch size)
_C.TRAIN.epoch_iters = 5000
# optimizer
_C.TRAIN.optim = "AdamW"
# momentum terms for Adam optimizer
_C.TRAIN.optim_beta = (0.5, 0.999)
# weight decay
_C.TRAIN.weight_decay = 0.0001
# epsilon term for Adam optimizer
_C.TRAIN.optim_eps = 1e-8
# indiviudal learning rate for encoder, generator and discriminator...

_C.TRAIN.warmup_steps = 20
_C.TRAIN.weight_decay = 0.001
_C.TRAIN.base_lr = 2e-4
_C.TRAIN.warmup_lr = 2e-7
_C.TRAIN.min_lr = 2e-5

# learning rate for the mapping network
_C.TRAIN.lr_MN = 2e-6
_C.TRAIN.lr_SE = 2e-4

# learning rate scheduler step size
_C.TRAIN.lr_scheduler = False

_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.name = 'cosine'
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.decay_epochs = 30
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.decay_rate = 0.1
# warmup_prefix used in CosineLRScheduler
_C.TRAIN.LR_SCHEDULER.warmup = True
# number of warmup epochs
_C.TRAIN.LR_SCHEDULER.warmup_epochs = 5
# miniumum learning rate
_C.TRAIN.LR_SCHEDULER.min_lr = 1e-6
# warmup learning rate
_C.TRAIN.LR_SCHEDULER.warmup_lr = 1e-8

# power in poly to drop LR
_C.TRAIN.lr_pow = 0.9
# fix bn params, only under finetuning
_C.TRAIN.fix_bn = False
# number of data loading workers
_C.TRAIN.workers = 20

# weights for losses
_C.TRAIN.w_GAN = 1.0
_C.TRAIN.w_StyleRecon = 10.0
_C.TRAIN.w_StyleDiv = 10.0
_C.TRAIN.lambda_StyleDiv = 1.0
_C.TRAIN.w_StyleDiv_iter = 100000
_C.TRAIN.w_NCE = 2.0
_C.TRAIN.w_Instance_NCE = 2.0
_C.TRAIN.w_Cycle = 2.0
_C.TRAIN.w_DClass = 1.0
_C.TRAIN.w_l_reg = 1.0
# maximum number of bounding boxes
_C.TRAIN.n_bbox = 4
# image normalization type one of imagenet, default, none
_C.TRAIN.img_norm = "default"
# frequency to save checkpoints
_C.TRAIN.save_epoch = 5
# manual seed
_C.TRAIN.seed = 304
# use cuda
_C.TRAIN.use_cuda = True
# IDs of GPUs to use (e.g. [0, 1, 2] | use -1 for CPU)
_C.TRAIN.gpu_ids = [0]
# output path for log, weights and intermediate results
_C.TRAIN.log_path = "./weights"
# distributed training
_C.TRAIN.distributed = False
# use label smoothing for discriminator
_C.TRAIN.smooth_label = False

# ------------------------------------------------------------------------------
# preprocessing options
# ------------------------------------------------------------------------------
_C.TRAIN.PREPROCESS = CN()
# resize image to a fixed size before training
_C.TRAIN.PREPROCESS.resize = True
# scale width of the image
_C.TRAIN.PREPROCESS.scale_width = False
# scale short side of the image
_C.TRAIN.PREPROCESS.scale_shortside = False
# crop and scale image to a fixed size
_C.TRAIN.PREPROCESS.zoom = False
# crop image to a fixed size
_C.TRAIN.PREPROCESS.crop = False
# crop image to a fixed size at a fixed location
_C.TRAIN.PREPROCESS.crop_size = [256, 256]
# crop image to a fixed size at a random location
_C.TRAIN.PREPROCESS.patch = False
# flip image horizontally
_C.TRAIN.PREPROCESS.flip = True
# trim image to a fixed size
_C.TRAIN.PREPROCESS.trim = False
# convert image to grayscale
_C.TRAIN.PREPROCESS.grayscale = False

# ------------------------------------------------------------------------------
# Visualization
# ------------------------------------------------------------------------------
_C.VISUAL = CN()
# -------------------------------------------------------------------------------
# enable visualization using visdom
_C.VISUAL.visdom = True
# display size in visdom
_C.VISUAL.display_winsize = 256
# number of columns to display in visdom
_C.VISUAL.display_ncols = 4
# window id of the web display
_C.VISUAL.display_id = -1
# visdom server
_C.VISUAL.server = "http://localhost"
# visdom port
_C.VISUAL.port = 8097
# name of the visdom workspace
_C.VISUAL.env = "main"
# frequency to save visualization results
_C.VISUAL.save_results_freq = 1000
# save intermediate training results to disk as html
_C.VISUAL.save_intermediate = True
# frequency to display training info
_C.VISUAL.display_sample_iter = 500
# frequency to save training images
_C.VISUAL.image_save_iter = 1000
# frequency to print training info on console
_C.VISUAL.print_losses_iter = 100
# frequency to display loss visualizations
_C.VISUAL.display_losses_iter = 200
# frequency to plot lrs in visdom
_C.VISUAL.plot_lr_iter = 500
# frequency to display lrs
_C.VISUAL.print_lrs_iter = 500

# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------
_C.VAL = CN()
# batch size currently only supports 1
_C.VAL.batch_size = 1
# output visualization during validation
_C.VAL.visualize = False
# the checkpoint to evaluate on
_C.VAL.checkpoint = "epoch_20.pth"

# -----------------------------------------------------------------------------
# Testing
# -----------------------------------------------------------------------------
_C.TEST = CN()
# dir
_C.TEST.dir = "data/test_images"
# currently only supports 1
_C.TEST.batch_size = 1
# the checkpoint to test on
_C.TEST.checkpoint = "epoch_20.pth"
# folder to output visualization results
_C.TEST.result = "./"
# epcch to load
_C.TEST.load_epoch = 20
# batch size
_C.TEST.batch_size = 1
# number of images
_C.TEST.num_images = 1000
# target_domains
_C.TEST.target_domains = ["summer", "winter", "spring", "autumn"]
# psi for style mixing
_C.TEST.psi = 1.0
# number of images per domain
_C.TEST.num_images_per_domain = 1
