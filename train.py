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

import utils
import initialize
import loss
from visualizer import Visualizer
# from collections import OrderedDict
from data import create_dataset
from util.config import cfg
from munch import Munch
from data_loader import MultiDomainDataset, InputProvider, RefProvider, get_train_loader, get_ref_loader

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

    logger.info('+ Loaded configuration file {}'.format(args.cfg))

    log_path_model = os.path.join(cfg.TRAIN.log_path, cfg.MODEL.name)
    if not os.path.isdir(log_path_model):
        os.makedirs(log_path_model)
    with open(os.path.join(log_path_model, 'config_{}.yaml'.format(cfg.MODEL.name)), 'w') as f:
        f.write("{}".format(cfg))

    device = torch.device('cuda:{}'.format(cfg.TRAIN.gpu_ids[0])) if cfg.TRAIN.gpu_ids else torch.device('cpu')

    # seed
    initialize.set_seed(cfg.TRAIN.seed)

    # data loader
    num_domains = len(cfg.DATASET.target_domain_names)

    data_loader = get_train_loader(
        img_size=cfg.MODEL.img_size,
        batch_size=cfg.TRAIN.batch_size_per_gpu,
        train_list=cfg.DATASET.train_list,
        target_domain_names=cfg.DATASET.target_domain_names
    )  # create a dataset given opt.dataset_mode and other options

    ref_list = []
    for td in cfg.DATASET.target_domain_names:
        if os.path.isdir(os.path.join(cfg.DATASET.ref_path, td)):
            ref_list.append(os.path.join(cfg.DATASET.ref_path, td))
        else:
            logger.warning("{} is not a directory".format(os.path.join(cfg.DATASET.ref_path, td)))
    assert len(ref_list) > 0, "Target domains {} not found in reference path {}".format(cfg.DATASET.target_domain_names,
                                                                                        cfg.DATASET.ref_path)
    if len(ref_list) < num_domains:
        logger.warning("Number of matching folders in the reference path {} is less than number of domains {}.".format(len(ref_list), num_domains))

    ref_loader = get_ref_loader(
        img_size=cfg.MODEL.img_size,
        batch_size=cfg.TRAIN.batch_size_per_gpu,
        ref_list=ref_list,
        target_domain_names=cfg.DATASET.target_domain_names,
        max_dataset_size=cfg.DATASET.max_dataset_size
    )

    input_provider = InputProvider(loader=data_loader, latent_dim=cfg.MODEL.latent_dim, num_domains=cfg.DATASET.num_domains)
    ref_provider = RefProvider(loader_ref=ref_loader)

    # model_load
    model_G, parameter_G, model_D, parameter_D, model_F, parameter_F, parameter_M = initialize.build_model(cfg=cfg, device=device, num_domains=num_domains,
                                                                                 distributed=cfg.TRAIN.distributed)

    # optimizer & scheduler
    optimizer_G = optim.Adam(parameter_G, float(cfg.TRAIN.lr_generator), betas=cfg.TRAIN.optim_beta)
    optimizer_M = optim.Adam(parameter_M, float(cfg.TRAIN.lr_mappingnetwork), betas=cfg.TRAIN.optim_beta)
    optimizer_D = optim.Adam(parameter_D, float(cfg.TRAIN.lr_discriminator), betas=cfg.TRAIN.optim_beta)
    # optimizer_F = optim.Adam(parameter_F, float(cfg.TRAIN.lr_MLP), betas=cfg.TRAIN.optim_beta)

    if cfg.MODEL.load_optimizer:
        logger.info('Loading Adam optimizer')
        optim_load_dict_g = torch.load(os.path.join(cfg.MODEL.weight_path, 'adam_g.pth'), map_location=device)
        optim_load_dict_d = torch.load(os.path.join(cfg.MODEL.weight_path, 'adam_g.pth'), map_location=device)
        optim_load_dict_f = torch.load(os.path.join(cfg.MODEL.weight_path, 'adam_g.pth'), map_location=device)
        optimizer_G.load_state_dict(optim_load_dict_g)
        optimizer_D.load_state_dict(optim_load_dict_d)
        # optimizer_F.load_state_dict(optim_load_dict_f)

    # TODO adjust lr for mapping network
    if cfg.TRAIN.lr_scheduler:
        lr_scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, cfg.TRAIN.scheduler_step_size, 0.1)
        lr_scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, cfg.TRAIN.scheduler_step_size, 0.1)
        # lr_scheduler_F = optim.lr_scheduler.StepLR(optimizer_F, cfg.TRAIN.scheduler_step_size, 0.1)

    criterions = initialize.set_criterions(cfg, device)

    visualizer = Visualizer(cfg.MODEL.name, cfg.TRAIN.log_path, cfg.VISDOM, cfg.DATASET.target_domain_names)

    # input_provider_val = InputProvider(data_loader.val, None, args.latent_dim, 'val')
    # inputs_val = next(input_provider_val)

    logger.info('+ Start Training')
    logger.info('++ Training for {} epoches with {} iterations per epoch'.format(cfg.TRAIN.end_epoch-cfg.TRAIN.start_epoch, cfg.TRAIN.epoch_iters//cfg.TRAIN.batch_size_per_gpu))    
    for epoch in range(cfg.TRAIN.start_epoch, cfg.TRAIN.end_epoch ):
        utils.model_mode(model_G, 1)
        utils.model_mode(model_D, 1)
        utils.model_mode(model_F, 1)
        visualizer.reset()  # save intermediate results to HTML at least once every epoch
        iter_date_time = time.time()

        dataset_size = len(data_loader)

        logger.info('++ Training progress: Epoch {}'.format(epoch + 1))

        box_feature = torch.empty(1).to(device)
        for i in range(0, cfg.TRAIN.epoch_iters//cfg.TRAIN.batch_size_per_gpu):
            inputs = next(input_provider)
            refs = next(ref_provider, inputs.d_src)
            # for key, val in inputs.items():
            #     if isinstance(val, torch.Tensor):
            #         inputs[key] = val.to(device)
            # for key, val in refs.items():
            #     if isinstance(val, torch.Tensor):
            #         refs[key] = val.to(device)


            # Model Forward
            fake_img, fake_box, features, d_src_pred = loss.model_forward_generation(inputs=inputs,
                                                                                     lat_trg = inputs.lat_trg,
                                                                                     refs = refs,
                                                                                     model=model_G,
                                                                                     n_bbox=cfg.DATASET.n_bbox,
                                                                                     feat_layers=cfg.MODEL.feat_layers)
            if cfg.TRAIN.w_StyleDiv > 0.0:
                fake_img_2, _, _, _ = loss.model_forward_generation(inputs=inputs, refs=refs,
                                                                    lat_trg=inputs.lat_trg_2,
                                                                    model=model_G,
                                                                    feat_layers=cfg.MODEL.feat_layers)
            else:
                fake_img_2 = torch.empty(0).to(device)
            recon_img, style_code, d_fake_pred = loss.model_forward_reconstruction(inputs=inputs, fake_img=fake_img,
                                                                      model=model_G, d_src_pred=d_src_pred,
                                                                      feat_layers=cfg.MODEL.feat_layers)
            if cfg.DATASET.n_bbox > 0 and len(features) > len(cfg.MODEL.feat_layers):
                features, box_feature = features[:-1], features[-1]

            # MLP_initialize
            if epoch == 0 and i == 0 and (
                    cfg.TRAIN.w_NCE != 0.0 or (cfg.TRAIN.w_Instance_NCE != 0.0 and cfg.TRAIN.data.n_bbox > 0)):
                if cfg.TRAIN.w_NCE != 0.0:
                    model_F.MLP_head.module.create_mlp(feats=features, device=device)
                if (cfg.TRAIN.w_Instance_NCE > 0.0 and cfg.DATASET.n_bbox > 0):
                    model_F.MLP_head_inst.module.create_mlp(feats=[box_feature], device=device)

                parameter_F = []
                for key, val in model_F.items():
                    model_F[key] = nn.DataParallel(val, device_ids=cfg.TRAIN.gpu_ids)
                    # model_F[key].to(device)
                    model_F[key].train()
                    parameter_F += list(val.parameters())
                optimizer_F = optim.Adam(parameter_F, lr=float(cfg.TRAIN.lr))

            # Backward & Optimizer
            optimize_start_time = time.time()
            # Discriminator
            utils.set_requires_grad(model_D.Discrim.module, True)
            optimizer_D.zero_grad()
            total_D_loss, D_losses = loss.compute_D_loss(inputs=inputs, 
                                                         refs=refs,  
                                                         fake_img=fake_img, 
                                                         model_D=model_D,
                                                         criterions=criterions)
            total_D_loss.backward()
            optimizer_D.step()

            # Generator
            fake_imgs = [fake_img, fake_img_2]
            utils.set_requires_grad(model_D.Discrim.module, False)
            optimizer_G.zero_grad()
            if (cfg.TRAIN.w_NCE > 0.0) or (cfg.TRAIN.w_Instance_NCE):
                optimizer_F.zero_grad()
            total_G_loss, G_losses = loss.compute_G_loss(inputs=inputs,
                                                         refs=refs,
                                                         fake_imgs=fake_imgs,
                                                         d_src_pred=d_src_pred,
                                                         recon_img=recon_img,
                                                         features=features,
                                                         box_feature=box_feature,
                                                         model_G=model_G,
                                                         model_D=model_D,
                                                         model_F=model_F,
                                                         criterions=criterions,
                                                         d_fake_img_pred=d_fake_pred,
                                                         cfg=cfg)
            total_G_loss.backward()
            optimizer_G.step()
            if (cfg.TRAIN.w_NCE > 0.0) or (cfg.TRAIN.w_Instance_NCE):
                optimizer_F.step()

            # Visualize(visdom)
            total_iters = epoch * cfg.TRAIN.epoch_iters + i
            # logging.info('++++ Training progress: total iters: {}'.format(total_iters)),
            losses = {}
            losses.update(G_losses)
            losses.update(D_losses)
            if (cfg.VISDOM.enabled):
                if (total_iters % cfg.TRAIN.display_losses_iter) == 0:
                    visualizer.plot_current_losses(epoch, float(i) / len(data_loader),
                                               {k: v.item() for k, v in losses.items()})
                if (total_iters % cfg.TRAIN.display_sample_iter) == 0:
                    current_visuals = {'input_img': inputs.img_src, 'generated_img_1': fake_img,
                                       'reference_img': refs.img_ref, 'reconstructed_img': recon_img}
                    current_domains = {'source_domain': inputs.d_src, 'predicted domain': d_src_pred, 'target_domain': refs.d_trg}
                    visualizer.display_current_samples(current_visuals, current_domains, epoch,
                                                       (total_iters % cfg.TRAIN.image_save_iter == 0))
            if (total_iters % cfg.TRAIN.print_losses_iter) == 0:
                visualizer.print_current_losses(epoch + 1, i, losses, time.time() - iter_date_time,
                                                optimize_start_time - iter_date_time)
            if (cfg.VISDOM.save_intermediate and total_iters % cfg.VISDOM.save_epoch_freq) == 0:
                utils.save_image_from_tensor(inputs.img_src, ncol=cfg.TRAIN.batch_size_per_gpu, filename= '{}/{}_source_image_ep_{}.jpg'.format(cfg.TRAIN.log_path, cfg.MODEL.name, str(epoch)))
                utils.save_image_from_tensor(fake_img, ncol=cfg.TRAIN.batch_size_per_gpu, filename= '{}/{}_fake_{}.jpg'.format(cfg.TRAIN.log_path, cfg.MODEL.name, str(epoch)))
                utils.save_image_from_tensor(recon_img.img_src, ncol=cfg.TRAIN.batch_size_per_gpu, filename= '{}/{}_recon_{}.jpg'.format(cfg.TRAIN.log_path, cfg.MODEL.name, str(epoch)))
                if cfg.TRAIN.w_StyleDiv > 0.0:
                    utils.save_image_from_tensor(fake_img_2, ncol=cfg.TRAIN.batch_size_per_gpu, filename= '{}/{}_fake2_{}.jpg'.format(cfg.TRAIN.log_path, cfg.MODEL.name, str(epoch)))                                                
            # Save model & optimizer and example images
        if epoch > 0 and (epoch % cfg.TRAIN.save_epoch) == 0:
            utils.save_component(cfg.TRAIN.log_path, cfg.MODEL.name, epoch, model_G, optimizer_G)
            utils.save_component(cfg.TRAIN.log_path, cfg.MODEL.name, epoch, model_D, optimizer_D)
            if (cfg.TRAIN.w_NCE > 0.0) or (cfg.TRAIN.w_Instance_NCE):
                utils.save_component(cfg.TRAIN.log_path, cfg.MODEL.name, epoch, model_F, optimizer_F)
