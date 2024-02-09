import copy
import logging
import os
import random
from typing import Any, Dict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel
from munch import Munch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import models
from models import blocks
import utils

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
        models.ContentEncoderV2(
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
    logging.info(">> Building the Transformer Encoder")
    TransformerEncoder = nn.DataParallel(
        models.Transformer(model_cfg=model_cfg, vis=False)
    )
    logging.info(">> Building the Transformer Generator")
    TransformerGenerator = nn.DataParallel(
        models.swin_generator.SwinGenerator(model_cfg=model_cfg)
    )
    logging.info(">> Building the MLP Blocks")
    MLPAdain = nn.DataParallel(
        blocks.MLP(
            input_dim=model_cfg.style_dim,
            output_dim=utils.get_num_adain_params(
                TransformerEncoder.module.transformer
            ),
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
    MappingNetwork_ema = copy.deepcopy(MappingNetwork)

    model = Munch(
        ContentEncoder=ContentEncoder,
        StyleEncoder=StyleEncoder,
        TransformerEnc=TransformerEncoder,
        MLPAdain=MLPAdain,        
        TransformerGen=TransformerGenerator,
        MappingNetwork=MappingNetwork,
        Discriminator=Discriminator,
        MLPHead=MLPHead,
        MLPHead2=MLPHead2,
        MLPHeadInst=MLPHeadInst,
    )

    model_ema = Munch(
        ContentEncoder=ContentEncoder_ema,
        StyleEncoder=StyleEncoder_ema,
        MappingNetwork=MappingNetwork_ema,
    )
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
