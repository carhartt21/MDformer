import os
import pdb
import time
import argparse
import logging

import yaml
# from dotmap import DotMap
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from model import StarFormer

import utils
import initialize
import loss
from visualizer import Visualizer
# from collections import OrderedDict
from data import create_dataset
from util.config import cfg
from munch import Munch
from data_loader import MultiDomainDataset, TrainProvider, RefProvider, get_train_loader, get_ref_loader
from torchvision.utils import draw_bounding_boxes
from PIL import Image
# TRAIN = 1
# EVAL = 0

parser = argparse.ArgumentParser(description='arguments yaml load')

args = parser.parse_args()

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='PyTorch Multi-Domain Image-to-Image Translation using Transformers'
    )
    parser.add_argument(
        '--cfg',
        default='config/config.yaml',
        metavar='FILE',
        help='path to config file',
        type=str,
    )
    parser.add_argument(
        '--gpu',
        default=0,
        help='gpu to use'
    )
    parser.add_argument(
        'opts',
        help='Modify config options using the command-line',
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    logger.info('====== Loaded configuration =======')
    logger.info(f'>> Source: {args.cfg}')
    logger.info(yaml.dump(cfg))
    # for k, v in cfg.items():
    #     logger.info('>> {} : {}'.format(k, v))
    log_path_model = os.path.join(cfg.TRAIN.log_path, cfg.MODEL.name)
    if not os.path.isdir(log_path_model):
        os.makedirs(log_path_model)
    with open(os.path.join(log_path_model, f'config_{cfg.MODEL.name}.yaml'), 'w') as f:
        f.write(f'{cfg}')

    device = torch.device(f'cuda:{cfg.TRAIN.gpu_ids[0]}' if cfg.TRAIN.gpu_ids else 'cpu')

    # seed
    initialize.set_seed(cfg.TRAIN.seed)

    # data loader
    num_domains = len(cfg.DATASET.target_domain_names)

     # create a dataset given opt.dataset_mode and other options

    train_list = []
    for td in cfg.DATASET.target_domain_names:
        if os.path.isdir(os.path.join(cfg.DATASET.train_dir, td)):
            train_list.append(os.path.join(cfg.DATASET.train_dir, td))
        else:
            logger.warning(f'{os.path.join(cfg.DATASET.train_dir, td)} is not a directory')
    assert len(train_list) > 0, f'Target domains {cfg.DATASET.target_domain_names} not found in train directory {cfg.DATASET.train_dir}'
    if len(train_list) < num_domains:
        logger.warning(f'Number of matching folders in the directory ({len(train_list)}) is less than number of domains {num_domains}.')

    ref_list = []
    for td in cfg.DATASET.target_domain_names:
        if os.path.isdir(os.path.join(cfg.DATASET.ref_dir, td)):
            ref_list.append(os.path.join(cfg.DATASET.ref_dir, td))
        else:
            logger.warning(f'{os.path.join(cfg.DATASET.ref_dir, td)} is not a directory')
    assert len(ref_list) > 0, f'Target domains {cfg.DATASET.target_domain_names} not found in reference directory {cfg.DATASET.ref_dir}'
    if len(ref_list) < num_domains:
        logger.warning(f'Number of matching folders in the directory {len(ref_list)} is less than number of domains {num_domains}.')


    data_loader = get_train_loader(
        img_size=cfg.MODEL.img_size,
        batch_size=cfg.TRAIN.batch_size_per_gpu,
        train_list=train_list,
        ref_list=ref_list,
        target_domain_names=cfg.DATASET.target_domain_names,
        normalize=cfg.TRAIN.img_norm
    )

    visualizer = Visualizer(cfg.MODEL.name, cfg.TRAIN.log_path, cfg.VISDOM, cfg.DATASET.target_domain_names)

    model = StarFormer(cfg=cfg, device=device, mode='train')
   
    model.train(data_loader, visualizer=visualizer)