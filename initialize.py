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

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # print(f"seed : {seed}")

def baseline_model_load(model_cfg, device):
    model_G = {}
    parameter_G = []
    model_D = {}
    parameter_D = []
    model_F = {}

    model_G['ContentEncoder'] = networks.ContentEncoder()
    model_G['StyleEncoder'] = networks.StyleEncoder()
    model_G['Transformer'] = networks.Transformer_Aggregator()
    model_G['MLP_Adain'] = networks.MLP()
    model_G['Generator'] = networks.Generator()

    model_D['Discrim'] = networks.NLayerDiscriminator()

    model_F['MLP_head'] = networks.MLP_Head()
    model_F['MLP_head_inst'] = networks.MLP_Head(nc=64)

    # model_M['Mapping Network'] = networks.MappingNetwork(num_domains=2, style_dim=64, hidden_dim=1024, num_layers=3)

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


def criterion_set(cfg, device):
    criterions = {}
    criterions['GAN'] = utils.GANLoss().to(device)
    criterions['Idt'] = torch.nn.L1Loss().to(device)
    criterions['NCE'] = utils.PatchNCELoss(cfg.TRAIN.batch_size_per_gpu).to(device)
    criterions['InstNCE'] = utils.PatchNCELoss(cfg.TRAIN.batch_size_per_gpu * cfg.DATASET.num_box).to(device)
    return criterions

def criterion_test(cfg, device):
    criterions = {}
    criterions['GAN'] = utils.GANLoss().to(device)
    criterions['Idt'] = torch.nn.L1Loss().to(device)
    criterions['NCE'] = utils.PatchNCELoss(cfg.TEST.batch_size).to(device)
    criterions['InstNCE'] = utils.PatchNCELoss(cfg.TEST.batch_size * cfg.DATASET.num_box).to(device)
    return criterions