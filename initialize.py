import random
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from collections import OrderedDict
# from lib.nn import user_scattered_collate, async_copy_to
from utils import user_scattered_collate

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from data.single_dataset import SingleDataset
# from data import create_dataset
# from data import TrainDataset
import utils
import networks
from typing import Any, Dict, List, Tuple, Union


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
    print("Seed set to: {}".format(seed))
    return

def build_model(model_cfg, device, num_domains=8, distributed=False):
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
    if distributed:
        print("Distributed Training")
        torch.cuda.set_device(model_cfg.gpu_ids[0])
        dist.init_process_group(backend='nccl', init_method='env://')
        device = torch.device('cuda:{}'.format(model_cfg.gpu_ids[0])) if model_cfg.gpu_ids else torch.device('cpu')
        print(f"device : {device}")
        print(f"model_cfg.gpu_ids : {model_cfg.gpu_ids}")

    model_G = {}
    parameter_G = []
    model_D = {}
    parameter_D = []
    model_F = {}


    # model_G['ContentEncoder'] = networks.ContentEncoder()
    # model_G['StyleEncoder'] = networks.StyleEncoder()
    # model_G['Transformer'] = networks.Transformer_Aggregator()
    # model_G['MLP_Adain'] = networks.MLP()
    # model_G['Generator'] = networks.Generator()
    
    model_G['ContentEncoder'] = nn.DataParallel(networks.ContentEncoder(input_channels=model_cfg.in_channels, n_generator_filters=model_cfg.n_generator_filters, n_downsampling=model_cfg.n_downsampling))
    
    model_G['StyleEncoder'] = nn.DataParallel(networks.StyleEncoder(input_channels=model_cfg.in_channels, 
                                                    n_generator_filters=model_cfg.n_generator_filters, 
                                                    style_dim=model_cfg.style_dim, 
                                                    num_domains= num_domains))
    
    model_G['Transformer'] = nn.DataParallel(networks.Transformer_Aggregator(input_size=model_cfg.img_size[0]//model_cfg.n_downsampling**2, 
                                                             patch_size=model_cfg.patch_size, 
                                                             embed_C=model_cfg.TRANSFORMER.embed_C, 
                                                             feat_C=model_cfg.n_generator_filters*2**(model_cfg.n_downsampling), 
                                                             depth=model_cfg.TRANSFORMER.depth, 
                                                             heads=model_cfg.TRANSFORMER.heads, 
                                                             mlp_dim=model_cfg.TRANSFORMER.mlp_dim))
    model_G['MLP_Adain'] = nn.DataParallel(networks.MLP(input_dim=model_cfg.style_dim, output_dim=2048))
    # model_G['MLP_Adain'] = nn.DataParallel(networks.MLP())

    model_G['DomainClassifier'] = nn.DataParallel(networks.TransformerClassifier(d_model=model_cfg.TRANSFORMER.embed_C, num_classes=num_domains))

    model_G['Generator'] = nn.DataParallel(networks.Generator(input_size=model_cfg.img_size[0]//model_cfg.n_downsampling**2, 
                                              patch_size=model_cfg.patch_size, 
                                              embed_C=model_cfg.TRANSFORMER.embed_C, 
                                              feat_C=model_cfg.TRANSFORMER.feat_C, 
                                              n_generator_filters=model_cfg.n_generator_filters,
                                              n_downsampling=model_cfg.n_downsampling,
                                              ))
    
    model_G['MappingNetwork'] = nn.DataParallel(networks.MappingNetwork(num_domains=num_domains, 
                                                            latent_dim=model_cfg.latent_dim,
                                                            style_dim=model_cfg.style_dim))
    

    model_D['Discrim'] = nn.DataParallel(networks.NLayerDiscriminator(input_channels=model_cfg.in_channels, ndf=model_cfg.n_discriminator_filters, n_layers=3))

    model_F['MLP_head'] = nn.DataParallel(networks.MLP_Head())
    model_F['MLP_head_inst'] = nn.DataParallel(networks.MLP_Head())


    if model_cfg.load_weight:
        print("Loading Network weights")
        for key in model_G.keys():
            # file = os.path.join(model_cfg.weight_path, f'{key}.pth')
            file = os.path.join(model_cfg.weight_path, f'{key}.pth')
            if os.path.isfile(file):
                print(f"Success load {key} weight")
                model_load_dict = torch.load(file, map_location=device)
                # breakpoint()
                keys = model_load_dict.keys()
                values = model_load_dict.values()

                new_keys = []
                for i, mykey in enumerate(keys):
                    # if i==len(keys)-1:
                    #     new_keys.append(key)
                    # else:
                    new_key = mykey[7:] #REMOVE 'module.'
                    new_keys.append(new_key)
                new_dict = OrderedDict(list(zip(new_keys,values)))
                # breakpoint()
                model_G[key].load_state_dict(new_dict)

                # model_G[key].load_state_dict(model_load_dict)
            else:
                print(f"Dose not exist {file}")

        for key in model_D.keys():
            file = os.path.join(model_cfg.weight_path, f'{key}.pth')
            if os.path.isfile(file):
                print(f"Success load {key} weight")
                model_load_dict = torch.load(file, map_location=device)
                keys = model_load_dict.keys()
                values = model_load_dict.values()

                new_keys = []
                for _key in keys:
                    new_key = _key[7:] #REMOVE 'module.'
                    new_keys.append(new_key)
                new_dict = OrderedDict(list(zip(new_keys,values)))
                # breakpoint()
                model_D[key].load_state_dict(new_dict)
                # model_D[key].load_state_dict(model_load_dict)
            else:
                print(f"Dose not exist {file}")
            
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
    
    # for key, val in model_F.items()
    #     model_F[key] = nn.DataParallel(val)
    #     model_F[key].to(device)
    #     model_F[key].train()

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
    criterions['InstNCE'] = utils.PatchNCELoss(cfg.TRAIN.batch_size_per_gpu * cfg.DATASET.n_bbox_max).to(device)
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
    criterions['InstNCE'] = utils.PatchNCELoss(cfg.TEST.batch_size * cfg.DATASET.n_bbox_max).to(device)
    criterions['Style_Div'] = torch.nn.L1Loss().to(device)
    criterions['Cycle'] = torch.nn.L1Loss().to(device)
    return criterions