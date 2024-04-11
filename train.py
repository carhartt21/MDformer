import os

import argparse
import logging
import builtins
from datetime import timedelta

import yaml

# from dotmap import DotMap
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import get_rank, get_world_size

from model import StarFormer

import initialize
from visualizer import Visualizer

# from collections import OrderedDict
from util.config import cfg
from data_loader import get_train_loader
from utils import synchronize, get_rank

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PyTorch Multi-Domain Image-to-Image Translation using Transformers"
    )
    parser.add_argument(
        "--cfg",
        default="config/config.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--gpu", default=0, help="gpu to use")
    
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    if cfg.TRAIN.distributed:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://", timeout=timedelta(0, 18000))
        synchronize()
        
    if cfg.TRAIN.distributed and get_rank() != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if get_rank() == 0:
        logger.info("===== Loaded configuration =====")
        logger.info(f">> Source: {args.cfg}")
        logger.info(yaml.dump(cfg))        
        log_path_model = os.path.join(cfg.TRAIN.log_path, cfg.MODEL.name)
        if not os.path.isdir(log_path_model):
            os.makedirs(log_path_model)
        with open(os.path.join(log_path_model, f"config_{cfg.MODEL.name}.yaml"), "w") as f:
            f.write(f"{cfg}")
            
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
    # seed
    initialize.set_seed(cfg.TRAIN.seed + get_rank())

    # data loader
    num_domains = len(cfg.DATASET.target_domain_names)

    train_list = []
    for td in cfg.DATASET.target_domain_names:
        if os.path.isdir(os.path.join(cfg.DATASET.train_dir, td)):
            train_list.append(os.path.join(cfg.DATASET.train_dir, td))
        else:
            logger.warning(
                f"{os.path.join(cfg.DATASET.train_dir, td)} is not a directory"
            )
    assert (
        len(train_list) > 0
    ), f"Target domains {cfg.DATASET.target_domain_names} not found in train directory {cfg.DATASET.train_dir}"
    if len(train_list) < num_domains:
        logger.warning(
            f"Number of matching folders in the directory ({len(train_list)}) is less than number of domains {num_domains}."
        )

    ref_list = []
    for td in cfg.DATASET.target_domain_names:
        if os.path.isdir(os.path.join(cfg.DATASET.ref_dir, td)):
            ref_list.append(os.path.join(cfg.DATASET.ref_dir, td))
        else:
            logger.warning(
                f"{os.path.join(cfg.DATASET.ref_dir, td)} is not a directory"
            )
    assert (
        len(ref_list) > 0
    ), f"Target domains {cfg.DATASET.target_domain_names} not found in reference directory {cfg.DATASET.ref_dir}"
    if len(ref_list) < num_domains:
        logger.warning(
            f"Number of matching folders in the directory {len(ref_list)} is less than number of domains {num_domains}."
        )

    data_loader = get_train_loader(
        img_size=cfg.MODEL.img_size,
        batch_size=cfg.TRAIN.batch_size_per_gpu,
        train_list=train_list,
        ref_list=ref_list,
        target_domain_names=cfg.DATASET.target_domain_names,
        normalize=cfg.TRAIN.img_norm,
        num_workers=cfg.TRAIN.num_workers,
        max_n_bbox=cfg.TRAIN.n_bbox,
        seg_threshold=0.8,  
        distributed=cfg.TRAIN.distributed,
    )

    if get_rank() == 0:
        visualizer = Visualizer(
            cfg.MODEL.name, cfg.TRAIN.log_path, cfg.VISUAL, cfg.DATASET.target_domain_names
        )   
    else:
        visualizer = None
        
    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = cfg.TRAIN.base_lr * cfg.TRAIN.batch_size_per_gpu * get_world_size() / 512.0
    linear_scaled_warmup_lr = cfg.TRAIN.LR_SCHEDULER.warmup_lr * cfg.TRAIN.batch_size_per_gpu  * get_world_size() / 512.0
    linear_scaled_min_lr = cfg.TRAIN.LR_SCHEDULER.min_lr * cfg.TRAIN.batch_size_per_gpu * get_world_size() / 512.0
    # gradient accumulation also need to scale the learning rate
    # cfg.defrost()
    # cfg.TRAIN.base_lr = linear_scaled_lr
    # cfg.TRAIN.warmup_lr = linear_scaled_warmup_lr
    # cfg.TRAIN.min_lr = linear_scaled_min_lr
    # cfg.freeze()

    model = StarFormer(cfg=cfg, mode="train", local_rank=local_rank, device=device)
    
    model.train(data_loader, visualizer=visualizer)
