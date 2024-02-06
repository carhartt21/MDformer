import random
import os
import logging
import copy
from munch import Munch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from collections import OrderedDict

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from data.single_dataset import SingleDataset
import utils
import models
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
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.info(">> Seed set to: {}".format(seed))
    return


def build_model(cfg):
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
    logging.info("===== Building the Model =====")
    # if distributed:
    #     logging.info(">> Distributed Training: {}".format(distributed))
    #     torch.cuda.set_device(device)
    #     dist.init_process_group(backend='nccl', init_method='env://')
    #     logging.info(">>>> Device : {}".format(device))
    #     # logging.info(f"model_cfg.gpu_ids : {model_cfg.gpu_ids}")
    model_cfg = cfg.MODEL
    num_domains = len(cfg.DATASET.target_domain_names)

    # domain_idxs = utils.get_domain_indexes(model_cfg.DATASET.target_domain_names)
    # logging.info("domain_idxs : {}".format(domain_idxs))
    # TODO: add test mode
    logging.info(">> Building the ContentEncoder")
    ContentEncoder = nn.DataParallel(
        models.ContentEncoder2(
            input_channels=model_cfg.in_channels,
            ngf=model_cfg.n_generator_filters,
            n_downsampling=model_cfg.n_downsampling,
        )
    )
    logging.info(">> Building the StyleEncoder")
    StyleEncoder = nn.DataParallel(
        models.StyleEncoder(
            input_channels=model_cfg.in_channels,
            n_generator_filters=model_cfg.n_generator_filters,
            style_dim=model_cfg.style_dim,
            num_domains=num_domains,
        )
    )
    logging.info(">> Building the Transformer")
    Transformer = nn.DataParallel(models.Transformer(model_cfg=model_cfg, vis=False))
    logging.info(">> Building the Generator")
    g_input_size = ((model_cfg.img_size[0] // model_cfg.n_downsampling**2)//model_cfg.patch_size)**2
    Generator = nn.DataParallel(
        models.Generator(
            input_size=model_cfg.img_size[0] // model_cfg.n_downsampling**2,
            patch_size=model_cfg.patch_size,
            embed_C=model_cfg.patch_embed_dim + model_cfg.sem_embed_dim,
            feat_C=g_input_size,
            n_generator_filters=model_cfg.n_generator_filters,
            n_downsampling=model_cfg.n_downsampling,
        )
    )
    logging.info(">> Building the MLP Block")
    # continue here
    MLPAdain = nn.DataParallel(
        models.MLP(
            input_dim=model_cfg.style_dim,
            output_dim=utils.get_num_adain_params(Transformer.module.transformer),
        )
    )

    logging.info(">> Building the Mapping Network")
    MappingNetwork = nn.DataParallel(
        models.MappingNetwork(
            num_domains=num_domains,
            latent_dim=model_cfg.latent_dim,
            style_dim=model_cfg.style_dim,
            hidden_dim=model_cfg.hidden_dim,
        )
    )

    logging.info(">> Building the Discriminator")
    Discriminator = nn.DataParallel(
        models.NLayerDiscriminator(
            input_channels=model_cfg.in_channels,
            ndf=model_cfg.n_discriminator_filters,
            n_layers=3,
            num_domains=num_domains,
        )
    )
    logging.info(">> Building the MLP Heads")
    if cfg.TRAIN.w_NCE > 0.0:
        MLPHead = nn.DataParallel(models.MLPHead())
        MLPHead2 = nn.DataParallel(models.NPMLPHead())
    if cfg.TRAIN.w_Instance_NCE > 0.0:
        MLPHeadInst = nn.DataParallel(models.MLPHead())

    ContentEncoder_ema = copy.deepcopy(ContentEncoder)
    StyleEncoder_ema = copy.deepcopy(StyleEncoder)
    # Transformer_ema = copy.deepcopy(Transformer)
    Generator_ema = copy.deepcopy(Generator)
    MappingNetwork_ema = copy.deepcopy(MappingNetwork)

    model = Munch(
        ContentEncoder=ContentEncoder,
        StyleEncoder=StyleEncoder,
        Transformer=Transformer,
        Generator=Generator,
        MLPAdain=MLPAdain,
        MappingNetwork=MappingNetwork,
        Discriminator=Discriminator,
        MLPHead=MLPHead,
        MLPHead2=MLPHead2,
        MLPHeadInst=MLPHeadInst,
    )

    model_ema = Munch(
        ContentEncoder=ContentEncoder_ema,
        StyleEncoder=StyleEncoder_ema,
        # Transformer=Transformer_ema,
        Generator=Generator_ema,
        MappingNetwork=MappingNetwork_ema,
    )

    # if model_cfg.load_weight:
    #     logging.info("+ Loading Network weights")
    #     for key in model_G.keys():
    #         file = os.path.join(model_cfg.weight_path, f'{key}.pth')
    #         if os.path.isfile(file):
    #             logging.info(">> Success load weight {}".format(key))
    #             model_load_dict = torch.load(file, map_location=device)
    #             keys = model_load_dict.keys()
    #             values = model_load_dict.values()

    #             new_keys = []
    #             for i, mykey in enumerate(keys):
    #                 new_key = mykey[7:] #REMOVE 'module.'
    #                 new_keys.append(new_key)
    #             new_dict = OrderedDict(list(zip(new_keys,values)))
    #             model_G[key].load_state_dict(new_dict)
    #         else:
    #             logging.info(">> Does not exist {}".format(file))

    #     for key in model_D.keys():
    #         file = os.path.join(model_cfg.weight_path, f'{key}.pth')
    #         if os.path.isfile(file):
    #             logging.info(">> Success load {} weight".format(key))
    #             model_load_dict = torch.load(file, map_location=device)
    #             keys = model_load_dict.keys()
    #             values = model_load_dict.values()

    #             new_keys = []
    #             for _key in keys:
    #                 new_key = _key[7:] #REMOVE 'module.'
    #                 new_keys.append(new_key)
    #             new_dict = OrderedDict(list(zip(new_keys,values)))
    #             model_D[key].load_state_dict(new_dict)
    #         else:
    #             logging.info(">> Does not exist {}".format(file))
    # for key, val in model.items():
    #     # if 'MLPHead' not in key:
    #     model[key] = nn.DataParallel(val)
    # for key, val in model_ema.items():
    # #     if 'MLPHead' not in key:
    #     model[key] = nn.DataParallel(val)
    #         model_ema[key] = nn.DataParallel(val)

    # for key, val in model_D.items():
    #     model_D[key] = nn.DataParallel(val)
    #     model_D[key].to(device)
    #     model_D[key].train()
    #     parameter_D += list(val.parameters())

    # for key, val in model_F.items():
    #     model_F[key] = nn.DataParallel(val)
    #     model_F[key].to(device)
    #     model_F[key].train()
    #     parameter_F += list(val.parameters())

    return model, model_ema


def set_criterions(cfg: Any, device: str) -> Dict[str, Any]:
    """
    Create a dictionary of critererions for different loss functions used in the model.

    Args:
        cfg (object): Configuration object containing various parameters.
        device (str): Device to be used for loss computation.

    Returns:
        dict: Dictionary containing different loss functions.
    """

    criterions = Munch()
    criterions.GAN = utils.GANLoss().to(device)
    criterions.Idt = torch.nn.L1Loss().to(device)
    criterions.NCE = utils.PatchNCELoss(cfg.TRAIN.batch_size_per_gpu).to(device)
    criterions.SemNCE = utils.SemNCELoss(cfg.TRAIN.batch_size_per_gpu).to(device)
    criterions.InstNCE = utils.PatchNCELoss(
        cfg.TRAIN.batch_size_per_gpu * cfg.TRAIN.n_bbox
    ).to(device)
    criterions.Style_Div = torch.nn.L1Loss().to(device)
    criterions.Cycle = torch.nn.L1Loss().to(device)
    criterions.DClass = torch.nn.CrossEntropyLoss(ignore_index=-1).to(device)
    return criterions
