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
_C.DATASET.name = 'OUTSIDE'
# dataset mode (e.g., single, aligned, unaligned)
_C.DATASET.dataset_mode = "unaligned"
# path to images (should have subfolders trainA, trainB, valA, valB, etc)
_C.DATASET.root_dataset = "./data/"
_C.DATASET.list_train = "./data/training.odgt"
_C.DATASET.list_val = "./data/validation.odgt"
# number of semantic classes 
_C.DATASET.num_seg_class = 150
# number of domains
_C.DATASET.num_domain = (3, 3, 3)
# TODO change from width and height
# maximum number of images per folder
_C.DATASET.max_dataset_size = 200000
# number of object boxes per image
_C.DATASET.num_box = -1
# enable random flip during training
_C.DATASET.random_flip = True
_C.DATASET.dir_A = "./data/trainA"
_C.DATASET.dir_B = "./data/trainB"

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# name of the model
_C.MODEL.name = "MDformer"
# load pretrained weights, False if training from scratch
_C.MODEL.load_weight = False
# load optimizer, False if training from scratch
_C.MODEL.load_optimizer = False
# path of pretrained weights. If training from scratch, set this as empty string
_C.MODEL.weight_path = ""
# weights to finetune net_decoder
# _C.MODEL.weights_decoder = ""
# number of feature channels between encoder and decoder
_C.MODEL.fc_dim = 2048
# feature layers for NCE loss
_C.MODEL.feat_layers = [0, 4, 8]
# number of patches
_C.MODEL.num_patches = 256
# patch size
_C.MODEL.patch_size = 1
# number of negative samples for NCE loss
_C.MODEL.num_neg = 64
# number of positive samples for NCE loss
_C.MODEL.num_pos = 1
# input image size
_C.MODEL.imgSize = (352, 352)



# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
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
_C.TRAIN.optim = "Adam"
# momentum terms for Adam optimizer
# TODO change from beta1, beta2 
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
_C.TRAIN.w_NCE = 2.0
_C.TRAIN.w_Instance_NCE = 2.0

# frequency to save checkpoints 
_C.TRAIN.save_epoch = 5
# frequency to display training info
_C.TRAIN.display_iter = 20
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
_C.TRAIN.log_path = "./weights"
# ------------------------------------------------------------------------------
# preprocessing options
# ------------------------------------------------------------------------------
_C.TRAIN.preprocess = CN()
# resize image to a fixed size before training
_C.TRAIN.preprocess.resize = True
# scale width of the image
_C.TRAIN.preprocess.scale_width = False
# scale short side of the image
_C.TRAIN.preprocess.scale_shortside = False
# crop and scale image to a fixed size
_C.TRAIN.preprocess.zoom = False
# crop image to a fixed size
_C.TRAIN.preprocess.crop = False
# crop image to a fixed size at a fixed location
_C.TRAIN.preprocess.crop_size = [256, 256]
# crop image to a fixed size at a random location
_C.TRAIN.preprocess.patch = False
# flip image horizontally
_C.TRAIN.preprocess.flip = True
# trim image to a fixed size
_C.TRAIN.preprocess.trim = False
# convert image to grayscale
_C.TRAIN.preprocess.grayscale = False

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
_C.VISDOM.server = "http://localhost"
# visdom port
_C.VISDOM.port = 8097
# name of the visdom workspace
_C.VISDOM.env = "main"
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
_C.VAL.checkpoint = "epoch_20.pth"

# -----------------------------------------------------------------------------
# Testing
# -----------------------------------------------------------------------------
_C.TEST = CN()
# currently only supports 1
_C.TEST.batch_size = 1
# the checkpoint to test on
_C.TEST.checkpoint = "epoch_20.pth"
# folder to output visualization results
_C.TEST.result = "./"