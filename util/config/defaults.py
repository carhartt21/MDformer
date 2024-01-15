from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
_C.DIR = './weights'

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
# dataset name
_C.DATASET = CN()
# dataset mode (e.g., single, aligned, unaligned, multi)
_C.DATASET.mode = 'multi'
# path to images (should have subfolders images, seg_labels, domain_labels)
_C.DATASET.root_dataset = './data/'
# txt file containing training image folder
_C.DATASET.train_list = './config/train_list.txt'
# path to the reference images
_C.DATASET.ref_path = './data/'
# List of target domains
_C.DATASET.target_domains = ['summer', 'winter', 'spring', 'autumn']
# number of semantic classes 
_C.DATASET.num_seg_class = 16
# number of domains
_C.DATASET.num_domains = 8
# maximum number of images per folder
_C.DATASET.max_dataset_size = 200000
# Maximum number of object boxes per image
_C.DATASET.n_bbox = -1
# enable random flip during training
_C.DATASET.random_flip = True

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# name of the model
_C.MODEL.name = 'MDformer'
# load pretrained weights, False if training from scratch
_C.MODEL.load_weight = False
# load optimizer, False if training from scratch
_C.MODEL.load_optimizer = False
# path of pretrained weights. If training from scratch, set this as empty string
_C.MODEL.weight_path = ''
# weights to finetune net_decoder
# _C.MODEL.weights_decoder = ''
# number of input channels
_C.MODEL.in_channels = 3
# number of feature channels between encoder and decoder
_C.MODEL.fc_dim = 2048
# feature layers for NCE loss
_C.MODEL.feat_layers = [0, 4, 8]
# number of patches for
_C.MODEL.num_patches = 256
# patch size
_C.MODEL.patch_size = 8
# number of negative samples for NCE loss
_C.MODEL.num_neg = 64
# number of positive samples for NCE loss
_C.MODEL.num_pos = 1
# input image size
_C.MODEL.img_size = (352, 352)
# Style code dimension
_C.MODEL.style_dim = 64
# Latent code dimension
_C.MODEL.latent_dim = 16
# dimension of the semantic embedding
_C.MODEL.sem_embed_dim = 64
# Mapping Network hidden dimension
_C.MODEL.hidden_dim = 1088
# number of content channels
_C.MODEL.content_dim = 256
# number of downsampling layers in the content encoder
_C.MODEL.n_downsampling = 2
# number of input filters of the generator
_C.MODEL.n_generator_filters = 64
# number of discriminator filters
_C.MODEL.n_discriminator_filters = 64
# ------------------------------------------------------------------------------
# Transformer parameters
# ------------------------------------------------------------------------------
_C.MODEL.TRANSFORMER = CN()
# number of input channels of the transformer
_C.MODEL.TRANSFORMER.embed_C = 1088
# number of feature channels of the transformer
_C.MODEL.TRANSFORMER.feat_C = 256
# number of transformer layers
_C.MODEL.TRANSFORMER.depth = 6
# number of heads in each transformer layer
_C.MODEL.TRANSFORMER.heads = 4
# dimension of each head
_C.MODEL.TRANSFORMER.mlp_dim = 4096



# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
# pretrain the classifier
_C.TRAIN.pretrain = False
# seed to sample training images
_C.TRAIN.seed_train = 304
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
_C.TRAIN.optim = 'Adam'
# momentum terms for Adam optimizer
_C.TRAIN.optim_beta = (0.5, 0.999)
# learning rate
_C.TRAIN.lr = 0.02
# indiviudal learning rate for encoder, generator and discriminator
_C.TRAIN.lr_encoder = 0.02
_C.TRAIN.lr_generator = 0.02
_C.TRAIN.lr_discriminator = 0.02
_C.TRAIN.lr_scheduler = True
# learning rate scheduler step size
_C.TRAIN.scheduler_step_size = 100
# power in poly to drop LR
_C.TRAIN.lr_pow = 0.9
# fix bn params, only under finetuning
_C.TRAIN.fix_bn = False
# number of data loading workers
_C.TRAIN.workers = 16

# weights for losses
_C.TRAIN.w_GAN = 1.0
_C.TRAIN.w_Recon = 10.0
_C.TRAIN.w_Style = 10.0
_C.TRAIN.w_Div = 10.0
_C.TRAIN.w_NCE = 2.0
_C.TRAIN.w_Instance_NCE = 2.0
_C.TRAIN.w_Cycle = 2.0

# frequency to save checkpoints 
_C.TRAIN.save_epoch = 5
# frequency to display training info
_C.TRAIN.display_iter = 500
# frequency to save training images
_C.TRAIN.image_save_iter = 200
# frequency to print training info on console
_C.TRAIN.print_freq = 100
# manual seed
_C.TRAIN.seed = 304
# use cuda
_C.TRAIN.use_cuda = True
# IDs of GPUs to use (e.g. [0, 1, 2] | use -1 for CPU)
_C.TRAIN.gpu_ids = [0]
# output path for log, weights and intermediate results
_C.TRAIN.log_path = './weights'
# distributed training
_C.TRAIN.distributed = False

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
_C.VISDOM = CN()
#-------------------------------------------------------------------------------
# enable visualization using visdom
_C.VISDOM.enabled = True
# display size in visdom
_C.VISDOM.display_winsize = 256
# frequency to display visualizations
_C.VISDOM.display_freq = 400
# number of columns to display in visdom
_C.VISDOM.display_ncols = 4
# window id of the web display
_C.VISDOM.display_id = -1
# visdom server
_C.VISDOM.server = 'http://localhost'
# visdom port
_C.VISDOM.port = 8097
# name of the visdom workspace
_C.VISDOM.env = 'main'
# frequency to save visualization results
_C.VISDOM.save_epoch_freq = 1000
# save intermediate training results to disk as html
_C.VISDOM.save_intermediate = True

# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------
_C.VAL = CN()
# batch size currently only supports 1
_C.VAL.batch_size = 1
# output visualization during validation
_C.VAL.visualize = False
# the checkpoint to evaluate on
_C.VAL.checkpoint = 'epoch_20.pth'

# -----------------------------------------------------------------------------
# Testing
# -----------------------------------------------------------------------------
_C.TEST = CN()
# currently only supports 1
_C.TEST.batch_size = 1
# the checkpoint to test on
_C.TEST.checkpoint = 'epoch_20.pth'
# folder to output visualization results
_C.TEST.result = './'