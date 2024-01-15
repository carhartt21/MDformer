import random
import os
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from collections import OrderedDict

from utils import user_scattered_collate

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from data.single_dataset import SingleDataset
import utils
import networks
from typing import Any, Dict, List, Tuple, Union
from torch.nn.parallel import DataParallel

logging.basicConfig(level=logging.INFO)

def set_seed(seed=42):
    """
    Set the random seed for reproducibility.

    Args:
        seed (int): The random seed value. Default is 42.
    """    
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.info("+ Seed set to: {}".format(seed))
    return

def build_model(model_cfg, device, distributed=False, num_domains=8):
    """
    Build the model for training.

    Args:
        model_cfg (object): The configuration object for the model.
        device (torch.device): The device to use for training.
        num_domains (int): The number of domains. Default is 8.
        distributed (bool): Whether to use distributed training. Default is False.

    Returns:
        tuple: A tuple containing the model_G, parameter_G, model_D, parameter_D, and model_F.
    """
    logging.info("+ Building the Model")
    logging.info("++ Distributed Training: {}".format(distributed))
    if distributed:
        torch.cuda.set_device(device)
        dist.init_process_group(backend='nccl', init_method='env://')
        logging.info("++ Device : {}".format(device))
        # logging.info(f"model_cfg.gpu_ids : {model_cfg.gpu_ids}")

    model_G = {}
    parameter_G = []
    model_D = {}
    parameter_D = []
    model_F = {}

    # domain_idxs = utils.get_domain_indexes(model_cfg.DATASET.target_domain_names)
    # logging.info("domain_idxs : {}".format(domain_idxs))

    logging.info("++ Building the ContentEncoder")
    model_G['ContentEncoder'] = DataParallel(networks.ContentEncoder(input_channels=model_cfg.in_channels, n_generator_filters=model_cfg.n_generator_filters, n_downsampling=model_cfg.n_downsampling))
    logging.info("++ Building the StyleEncoder")
    model_G['StyleEncoder'] = DataParallel(networks.StyleEncoder(input_channels=model_cfg.in_channels, 
                                                        n_generator_filters=model_cfg.n_generator_filters, 
                                                        style_dim=model_cfg.style_dim, 
                                                        num_domains=num_domains))
    logging.info("++ Building the Transformer")
    model_G['Transformer'] = DataParallel(networks.Transformer_Aggregator(input_size=model_cfg.img_size[0]//model_cfg.n_downsampling**2, 
                                                                 patch_size=model_cfg.patch_size, 
                                                                 patch_embed_C=model_cfg.TRANSFORMER.embed_C, 
                                                                 sem_embed_C=model_cfg.sem_embed_dim,
                                                                 feat_C=model_cfg.n_generator_filters*2**(model_cfg.n_downsampling), 
                                                                 depth=model_cfg.TRANSFORMER.depth, 
                                                                 heads=model_cfg.TRANSFORMER.heads, 
                                                                 mlp_dim=model_cfg.TRANSFORMER.mlp_dim))
    logging.info("++ Building the DomainClassifier")
    model_G['DomainClassifier'] = DataParallel(networks.TransformerClassifier(input_dim=model_cfg.TRANSFORMER.embed_C + model_cfg.sem_embed_dim, num_classes=num_domains))
    logging.info("++ Building the Generator")
    model_G['Generator'] = DataParallel(networks.Generator(input_size=model_cfg.img_size[0]//model_cfg.n_downsampling**2, 
                                                  patch_size=model_cfg.patch_size, 
                                                  embed_C=model_cfg.TRANSFORMER.embed_C + model_cfg.sem_embed_dim, 
                                                  feat_C=model_cfg.TRANSFORMER.feat_C, 
                                                  n_generator_filters=model_cfg.n_generator_filters,
                                                  n_downsampling=model_cfg.n_downsampling))
    logging.info("++ Building the MLP Block") 
    model_G['MLP_Adain'] = DataParallel(networks.MLP(input_dim=model_cfg.style_dim, output_dim=2176))

    logging.info("++ Building the Mapping Network")
    model_G['MappingNetwork'] = DataParallel(networks.MappingNetwork(num_domains=num_domains, 
                                                            latent_dim=model_cfg.latent_dim,
                                                            style_dim=model_cfg.style_dim))

    logging.info("++ Building the Discriminator")
    model_D['Discrim'] = DataParallel(networks.NLayerDiscriminator(input_channels=model_cfg.in_channels, ndf=model_cfg.n_discriminator_filters, n_layers=3, num_domains=num_domains))
    logging.info("++ Building the MLP Heads for Loss Calculation")
    model_F['MLP_head'] = DataParallel(networks.MLP_Head())
    model_F['MLP_head_inst'] = DataParallel(networks.MLP_Head())

    # model_G['ContentEncoder'] = DDP(networks.ContentEncoder(input_channels=model_cfg.in_channels, n_generator_filters=model_cfg.n_generator_filters, n_downsampling=model_cfg.n_downsampling))
    
    # model_G['StyleEncoder'] = DDP(networks.StyleEncoder(input_channels=model_cfg.in_channels, 
    #                                                 n_generator_filters=model_cfg.n_generator_filters, 
    #                                                 style_dim=model_cfg.style_dim, 
    #                                                 num_domains= num_domains))
    
    # model_G['Transformer'] = DDP(networks.Transformer_Aggregator(input_size=model_cfg.img_size[0]//model_cfg.n_downsampling**2, 
    #                                                          patch_size=model_cfg.patch_size, 
    #                                                          embed_C=model_cfg.TRANSFORMER.embed_C, 
    #                                                          feat_C=model_cfg.n_generator_filters*2**(model_cfg.n_downsampling), 
    #                                                          depth=model_cfg.TRANSFORMER.depth, 
    #                                                          heads=model_cfg.TRANSFORMER.heads, 
    #                                                          mlp_dim=model_cfg.TRANSFORMER.mlp_dim))
    # model_G['MLP_Adain'] = DDP(networks.MLP(input_dim=model_cfg.style_dim, output_dim=2048))

    # model_G['DomainClassifier'] = DDP(networks.TransformerClassifier(input_dim=model_cfg.TRANSFORMER.embed_C, num_classes=num_domains))

    # model_G['Generator'] = DDP(networks.Generator(input_size=model_cfg.img_size[0]//model_cfg.n_downsampling**2, 
    #                                           patch_size=model_cfg.patch_size, 
    #                                           embed_C=model_cfg.TRANSFORMER.embed_C, 
    #                                           feat_C=model_cfg.TRANSFORMER.feat_C, 
    #                                           n_generator_filters=model_cfg.n_generator_filters,
    #                                           n_downsampling=model_cfg.n_downsampling,
    #                                           ))
    
    # model_G['MappingNetwork'] = DDP(networks.MappingNetwork(num_domains=num_domains, 
    #                                                         latent_dim=model_cfg.latent_dim,
    #                                                         style_dim=model_cfg.style_dim))
    

    # model_D['Discrim'] = DDP(networks.NLayerDiscriminator(input_channels=model_cfg.in_channels, ndf=model_cfg.n_discriminator_filters, n_layers=3, num_domains=num_domains))

    # model_F['MLP_head'] = DDP(networks.MLP_Head())
    # model_F['MLP_head_inst'] = DDP(networks.MLP_Head())

    if model_cfg.load_weight:
        logging.info("+ Loading Network weights")
        for key in model_G.keys():
            file = os.path.join(model_cfg.weight_path, f'{key}.pth')
            if os.path.isfile(file):
                logging.info("++ Success load weight {}".format(key))
                model_load_dict = torch.load(file, map_location=device)
                keys = model_load_dict.keys()
                values = model_load_dict.values()

                new_keys = []
                for i, mykey in enumerate(keys):
                    new_key = mykey[7:] #REMOVE 'module.'
                    new_keys.append(new_key)
                new_dict = OrderedDict(list(zip(new_keys,values)))
                model_G[key].load_state_dict(new_dict)
            else:
                logging.info("++ Does not exist {}".format(file))

        for key in model_D.keys():
            file = os.path.join(model_cfg.weight_path, f'{key}.pth')
            if os.path.isfile(file):
                logging.info("++ Success load {} weight".format(key))
                model_load_dict = torch.load(file, map_location=device)
                keys = model_load_dict.keys()
                values = model_load_dict.values()

                new_keys = []
                for _key in keys:
                    new_key = _key[7:] #REMOVE 'module.'
                    new_keys.append(new_key)
                new_dict = OrderedDict(list(zip(new_keys,values)))
                model_D[key].load_state_dict(new_dict)
            else:
                logging.info("++ Does not exist {}".format(file))
            
    for key, val in model_G.items():
        model_G[key] = nn.DataParallel(val)
        model_G[key].to(device)
        model_G[key].train()
        parameter_G += list(val.parameters())

    for key, val in model_D.items():
        model_D[key] = nn.DataParallel(val)
        model_D[key].to(device)
        model_D[key].train()
        parameter_D += list(val.parameters())
    
    for key, val in model_F.items():
        model_F[key] = nn.DataParallel(val)
        model_F[key].to(device)
        model_F[key].train()

    return model_G, parameter_G, model_D, parameter_D, model_F


def set_criterions(cfg: Any, device: str) -> Dict[str, Any]:
    """
    Create a dictionary of critererions for different loss functions used in the model.

    Args:
        cfg (object): Configuration object containing various parameters.
        device (str): Device to be used for loss computation.

    Returns:
        dict: Dictionary containing different loss functions.
    """

    criterions: Dict[str, Any] = {}
    criterions['GAN'] = utils.GANLoss().to(device)
    criterions['Idt'] = torch.nn.L1Loss().to(device)
    criterions['NCE'] = utils.PatchNCELoss(cfg.TRAIN.batch_size_per_gpu).to(device)
    criterions['InstNCE'] = utils.PatchNCELoss(cfg.TRAIN.batch_size_per_gpu * cfg.DATASET.n_bbox).to(device)
    criterions['Style_Div'] = torch.nn.L1Loss().to(device)
    criterions['Cycle'] = torch.nn.L1Loss().to(device)
    return criterions

def criterion_test(cfg: Any, device: str) -> Dict[str, Any]:
    """
    Create a dictionary of critererions for different loss functions used in the model.

    Args:
        cfg (object): Configuration object containing various parameters.
        device (str): Device to be used for loss computation.

    Returns:
        dict: Dictionary containing different loss functions.
    """
    criterions: Dict[str, Any] = {}
    criterions['GAN'] = utils.GANLoss().to(device)
    criterions['Idt'] = torch.nn.L1Loss().to(device)
    criterions['NCE'] = utils.PatchNCELoss(cfg.TEST.batch_size).to(device)
    criterions['InstNCE'] = utils.PatchNCELoss(cfg.TEST.batch_size * cfg.DATASET.n_bbox).to(device)
    criterions['Style_Div'] = torch.nn.L1Loss().to(device)
    criterions['Cycle'] = torch.nn.L1Loss().to(device)
    return criterions