import os
import pdb
import time
import argparse

import yaml
from dotmap import DotMap
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import utils
import initialize
import loss
from visualizer import Visualizer
from collections import OrderedDict
from data import create_dataset
from util.config import cfg

from data_loader import MultiDomainDataset, InputProvider, get_train_loader

TRAIN = 1
EVAL  = 0

parser = argparse.ArgumentParser(description='arguments yaml load')
parser.add_argument("--conf",
                    type=str,
                    help="configuration file path",
                    default="./config/base_train.yaml")

args = parser.parse_args()


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

    logger = utils.setup_logger(distributed_rank=0)   
    logger.info('Loaded configuration file {}'.format(args.cfg))

    # train_cfg = DotMap(conf['Train'])
    device = torch.device('cuda:{}'.format(cfg.TRAIN.gpu_ids[0])) if cfg.TRAIN.gpu_ids else torch.device('cpu')        

    # seed 
    initialize.set_seed(cfg.TRAIN.seed)

    # data loader
    # continue here
    data_loader = get_train_loader(root='', which=cfg.DATASET.mode, img_size=cfg.MODEL.img_size, batch_size=cfg.TRAIN.batch_size_per_gpu, train_list=cfg.TRAIN.train_list)  # create a dataset given opt.dataset_mode and other options
    # ref_loader = get_train_loader(root=cfg.ref_img_dir, which='reference', img_size=args.img_size, img_size=cfg.MODEL.img_size, batch_size=cfg.TRAIN.batch_size_per_gpu)
    input_provider = InputProvider(loader=data_loader, latent_dim=cfg.MODEL.latent_dim, mode='train', num_domains=cfg.DATASET.num_domains)

    #model_load
    model_G, parameter_G, model_D, parameter_D, model_F = initialize.build_model(cfg.MODEL, device, cfg.DATASET.num_domains, cfg.TRAIN.distributed)

    # optimizer & scheduler
    optimizer_G = optim.Adam(parameter_G, float(cfg.TRAIN.lr_generator),betas=cfg.TRAIN.optim_beta)
    optimizer_D = optim.Adam(parameter_D, float(cfg.TRAIN.lr_discriminator),betas=cfg.TRAIN.optim_beta)

    if cfg.MODEL.load_optimizer:
        print('Loading Adam optimizer')
        optim_load_dict_g = torch.load(os.path.join(cfg.MODEL.weight_path,'adam_g.pth'), map_location=device)
        optim_load_dict_d = torch.load(os.path.join(cfg.MODEL.weight_path,'adam_g.pth'), map_location=device)
        optim_load_dict_f = torch.load(os.path.join(cfg.MODEL.weight_path,'adam_g.pth'), map_location=device)
        optimizer_G.load_state_dict(optim_load_dict_g)
        optimizer_D.load_state_dict(optim_load_dict_d)
        
        # optimizer_F.load_state_dict(optim_load_dict)
    # TODO maybe adjust lr for mapping network
    if cfg.TRAIN.lr_scheduler:
        lr_scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, cfg.TRAIN.scheduler_step_size, 0.1)
        lr_scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, cfg.TRAIN.scheduler_step_size, 0.1)

    criterions = initialize.set_criterions(cfg, device)

    visualizer = Visualizer(cfg.MODEL.name, cfg.TRAIN.log_path, cfg.VISDOM)

    # input_provider_val = InputProvider(data_loader.val, None, args.latent_dim, 'val')
    # inputs_val = next(input_provider_val)

    print('Start Training')
    for epoch in range(cfg.TRAIN.start_epoch, cfg.TRAIN.end_epoch):

        utils.model_mode(model_G,TRAIN)
        utils.model_mode(model_D,TRAIN)
        utils.model_mode(model_F,TRAIN)
        visualizer.reset() # save intermediate results to HTML at least once every epoch
        iter_date_time = time.time()

        dataset_size = len(data_loader)

        print(f'Training progress(ep:{epoch+1})')

        box_feature = torch.empty(1).to(device)  
        # print('inputs  {} {}'.format(len(inputs), inputs))
        # inputs = inputs[0]
        for i in range(0, cfg.TRAIN.epoch_iters, cfg.TRAIN.batch_size_per_gpu):
            inputs = next(input_provider)            
            for key, val in inputs.items():
                if isinstance(val, torch.Tensor):
                    # print('key: {} val: {}'.format(key, val.shape))
                    inputs[key] = val.to(device)

            # Model Forward
            fake_img, fake_box, features, y_src_pred = loss.model_forward_generation(inputs=inputs, z_trg=inputs.z_trg, model=model_G, n_bbox=cfg.DATASET.n_bbox_max, feat_layers=cfg.MODEL.feat_layers)
            if cfg.TRAIN.w_Div > 0.0:
                fake_img_2, _, _, _= loss.model_forward_generation(inputs=inputs, z_trg=inputs.z_trg2, model=model_G,feat_layers=cfg.MODEL.feat_layers)
            else:
                fake_img_2 = torch.empty(1).to(device)
            recon_img, style_code = loss.model_forward_reconstruction(inputs=inputs, fake_img=fake_img, model=model_G, y_src_pred=y_src_pred, feat_layers=cfg.MODEL.feat_layers)
            if cfg.DATASET.n_bbox_max > 0 and len(features) > len(cfg.MODEL.feat_layers):
                features, box_feature =  features[:-1], features[-1]

            # MLP_initialize
            if epoch == 0 and i == 0 and (cfg.TRAIN.w_NCE != 0.0  or (cfg.TRAIN.w_Instance_NCE != 0.0 and cfg.TRAIN.data.n_bbox > 0)):
                if cfg.TRAIN.w_NCE != 0.0:
                    model_F['MLP_head'].module.module.create_mlp(feats=features, device=device)
                if (cfg.TRAIN.w_Instance_NCE > 0.0 and cfg.DATASET.n_bbox_max > 0):
                    model_F['MLP_head_inst'].create_mlp(feats=[box_feature], device=device)

                parameter_F = []
                for key, val in model_F.items():
                    model_F[key] = nn.DataParallel(val, device_ids=cfg.TRAIN.gpu_ids)
                    # model_F[key].to(device)
                    model_F[key].train()
                    parameter_F += list(val.parameters())
                optimizer_F = optim.Adam(parameter_F, lr=float(cfg.TRAIN.lr))

            #Backward & Optimizer
            optimize_start_time = time.time() 
            fake_imgs = [fake_img, fake_img_2]
            #Discriminator  
            utils.set_requires_grad(model_D['Discrim'].module, True)
            optimizer_D.zero_grad()
            total_D_loss, D_losses = loss.compute_D_loss(inputs=inputs, fake_img=fake_img, model_D=model_D, criterions=criterions)
            total_D_loss.backward()
            optimizer_D.step()

            #Generator                         
            utils.set_requires_grad(model_D['Discrim'].module, False)
            optimizer_G.zero_grad()
            optimizer_F.zero_grad()
            total_G_loss, G_losses = loss.compute_G_loss(inputs, fake_imgs, recon_img, style_code, features, box_feature, model_G, model_D, model_F, criterions, cfg)
            total_G_loss.backward()
            optimizer_G.step()
            optimizer_F.step()

            #Visualize(visdom)
            total_iters = epoch * len(data_loader) + (i+1)
            losses = {};  losses.update(G_losses);  losses.update(D_losses) 
            if (cfg.VISDOM.enabled):
                visualizer.plot_current_losses(epoch, float(i) / len(data_loader), {k: v.item() for k, v in losses.items()})
                if (total_iters % cfg.TRAIN.display_iter) == 0:
                    current_visuals = {'real_img':inputs['Source'], 'fake_img':fake_img, 'style_img':inputs['Target'], 'recon_img':recon_img}
                    visualizer.display_current_results(current_visuals, epoch,  (total_iters % cfg.TRAIN.image_save_iter == 0))
            else:
                if (total_iters % cfg.TRAIN.print_freq) == 0: 
                    visualizer.print_current_losses(epoch, i, losses, time.time() - iter_date_time, optimize_start_time - iter_date_time)   
                # Save model & optimizer            
            if epoch > 0 and (epoch % cfg.TRAIN.save_epoch) == 0:
                utils.save_component(cfg.TRAIN.log_path, cfg.MODEL.name, epoch, model_G, optimizer_G)
                utils.save_component(cfg.TRAIN.log_path, cfg.MODEL.name, epoch, model_D, optimizer_D)
                utils.save_component(cfg.TRAIN.log_path, cfg.MODEL.name, epoch, model_F, optimizer_F)

            # utils.save_color(inputs['A'], 'test/realA', str(epoch))
            # utils.save_color(inputs['B'], 'test/realB', str(epoch))
            # utils.save_color(fake_img, 'test/fake', str(epoch))
            # utils.save_color(recon_img, 'test/recon', str(epoch))
