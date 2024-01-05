import torch
import torchvision
import torch.nn as nn
import numpy as np

import utils
import pdb
from typing import List, Tuple, Union
 

######################################### Forward #########################################

def model_forward_generation(inputs: Union[torch.Tensor, dict], model: dict, n_bbox: int = -1, feat_layers: List[str] = []) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # when performing generation, the input is the content image from the data loader
    feat_content, features = model['ContentEncoder'](inputs.img, feat_layers)
    style = model['MappingNetwork'](inputs.target_domain)
    utils.assign_adain_params(model['MLP_Adain'](style), model['Transformer'].module.module.transformer.layers)
    if 'BBox' in inputs:
        features += [model['Transformer'].module.extract_box_feature(feat_content, inputs.BBox, n_bbox)]

    if n_bbox == -1:
        aggregated_feat, _ = model['Transformer'](feat_content)
    else:
        aggregated_feat, aggregated_box = model['Transformer'](feat_content, inputs.BBox, n_bbox)        

    fake = model['Generator'](aggregated_feat)
    fake_box = model['Generator'](aggregated_box, inputs['A_box']) if 'A_box' in inputs else None
    
    return fake, fake_box, features

def model_forward_reconstruction(fake_image: torch.Tensor, model: dict, source_domain: torch.Tensor, n_bbox: int = -1, bboxes: torch.Tensor=None, feat_layers: List[str] = []) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    feat_content, _ = model['ContentEncoder'](fake_image, feat_layers)
    style_code = model['StyleEncoder'](source_domain)
    utils.assign_adain_params(model['MLP_Adain'](style_code), model['Transformer'].module.module.transformer.layers)

    if n_bbox == -1:
        aggregated_feat, _ = model['Transformer'](feat_content)
    else:
        aggregated_feat, aggregated_box = model['Transformer'](feat_content, bboxes, n_bbox)

    rec_img = model['Generator'](aggregated_feat)
    rec_box = model['Generator'](aggregated_box, bboxes) if bboxes else None
    
    return rec_img, rec_box, style_code

######################################### Total Loss #########################################

def compute_D_loss(inputs, fake_img, model_D, criterions):
    """Calculate GAN loss for the discriminator"""
    D_losses = {}    
    D_losses['D_fake_loss'] = compute_Discrim_loss(fake_img, model_D['Discrim'], criterions['GAN'], False) 
    D_losses['D_real_loss'] = compute_Discrim_loss(inputs['Target'], model_D['Discrim'], criterions['GAN'], True) 
    total_D_loss = (D_losses['D_fake_loss'] +  D_losses['D_real_loss']) * 0.5

    return total_D_loss, D_losses

def compute_G_loss(inputs, fake_img, recon_img, style_code, features, box_feature, model_G, model_D, model_F, criterions, cfg):
    """Calculate loss for the generator"""
    G_losses = {}
    total_G_loss = 0

    if cfg.TRAIN.w_GAN > 0.0:
        G_losses['GAN_loss'] = cfg.TRAIN.w_GAN * compute_GAN_loss(fake_img, model_D['Discrim'], criterions['GAN'])
    if cfg.TRAIN.w_Recon > 0.0:
        G_losses['recon_loss'] = cfg.TRAIN.w_Recon * compute_style_recon_loss(recon_img, inputs['Target'], criterions['Idt'])
    if cfg.TRAIN.w_Style > 0.0:
        recon_style_code = model_G['StyleEncoder'](recon_img)
        G_losses['style_loss'] = cfg.TRAIN.w_Style * compute_style_recon_loss(recon_style_code, style_code, criterions['Idt'])
    
    if cfg.TRAIN.w_NCE > 0.0 or (cfg.TRAIN.w_Instance_NCE > 0.0 and cfg.DATASET.n_bbox > 0):
        fake_feat_content, fake_features = model_G['ContentEncoder'](fake_img, cfg.MODEL.feat_layers)

        if cfg.TRAIN.w_Instance_NCE > 0.0 and cfg.DATASET.n_bbox > 0:
            fake_box_feature = model_G['Transformer'].module.extract_box_feature(fake_feat_content, inputs['A_box'], cfg.DATASET.n_bbox)

    if cfg.TRAIN.w_NCE > 0.0:
        G_losses['NCE_loss'] = cfg.TRAIN.w_NCE * compute_NCE_loss(fake_features, features, model_F['MLP_head'], criterions['NCE'], cfg.MODEL.num_patches)
    if cfg.TRAIN.w_Instance_NCE > 0.0 and cfg.DATASET.n_bbox > 0:
        valid_box=torch.where(inputs['A_box'][:,:,0] != -1, True,False).view(-1)
        if valid_box[valid_box ==  True].shape[0] == 0.0:
            G_losses['instNCE_loss'] = torch.tensor(0.0).to(inputs['A'].device)
        else:
            criterions['InstNCE'].batch_size = valid_box[valid_box ==  True].shape[0]
            G_losses['instNCE_loss'] = cfg.TRAIN.w_Instance_NCE * compute_NCE_loss([fake_box_feature[valid_box,:,:,:]], [box_feature[valid_box,:,:,:]], model_F['MLP_head_inst'], criterions['InstNCE'], 64)
    if cfg.TRAIN.w_Style_Div > 0.0:
        G_losses['style_div_loss'] = cfg.TRAIN.w_Style_Div * compute_style_diversification_loss(fake_img, recon_img, criterions['Style_Div'])
    
    if cfg.TRAIN.w_Cycle > 0.0:
        G_losses['cycle_loss'] = cfg.TRAIN.w_Cycle * compute_cycle_loss(inputs['A'], recon_img, criterions['Cycle'])

    for loss in G_losses.values():
        total_G_loss += loss

    return total_G_loss, G_losses

######################################### Each Loss #########################################
def compute_Discrim_loss(img, model, criterion, Target=True):
    """
    Calculate the discriminator loss.

    Args:
        img (torch.Tensor): Input image.
        model (torch.nn.Module): Discriminator model.
        criterion (torch.nn.Module): Criterion for calculating the loss.
        Target (bool, optional): Whether the target is real or fake. Defaults to True.

    Returns:
        torch.Tensor: Discriminator loss.
    """
    pred = model(img.detach())
    return criterion(pred, Target).mean()


def compute_GAN_loss(fake, model, criterion):
    """
    Calculate the GAN loss.

    Args:
        fake (torch.Tensor): Fake image.
        model (torch.nn.Module): Discriminator model.
        criterion (torch.nn.Module): Criterion for calculating the loss.

    Returns:
        torch.Tensor: GAN loss.
    """
    pred_fake = model(fake)
    return criterion(pred_fake, True).mean()


def compute_style_recon_loss(src, tgt, criterion): 
    """
    Calculate the style reconstruction loss.

    Args:
        src (torch.Tensor): Source image.
        tgt (torch.Tensor): Target image.
        criterion (torch.nn.Module): Criterion for calculating the loss.

    Returns:
        torch.Tensor: Style reconstruction loss.
    """
    return criterion(src, tgt)


def compute_NCE_loss(feat_q, feat_k, model, criterionNCE, num_patches): 
    """
    Calculate the NCE (Normalized Cross Entropy) loss.

    Args:
        feat_q (List[torch.Tensor]): Query features.
        feat_k (List[torch.Tensor]): Key features.
        model (torch.nn.Module): MLP model.
        criterionNCE (torch.nn.Module): Criterion for calculating the loss.
        num_patches (int): Number of patches.

    Returns:
        torch.Tensor: NCE loss.
    """
    feat_k_pool, sample_ids = model(feat_k, num_patches, None)
    feat_q_pool, _ = model(feat_q, num_patches, sample_ids)

    total_nce_loss = 0.0    
    for f_q, f_k in zip(feat_q_pool, feat_k_pool):
        total_nce_loss += criterionNCE(f_q, f_k).mean()

    return total_nce_loss / len(feat_q)


def compute_style_diversification_loss(img1, img2, criterion):
    """
    Calculate the style diversification loss.

    Args:
        img1 (torch.Tensor): First generated image.
        img2 (torch.Tensor): Second generated image.
        criterion (torch.nn.Module): Criterion for calculating the loss.

    Returns:
        torch.Tensor: Style diversification loss.
    """
    return criterion(img1, img2)


def compute_cycle_loss(img, rec_img, criterion):
    """
    Calculate the cycle consistency loss.

    Args:
        img (torch.Tensor): Original image.
        rec_img (torch.Tensor): Reconstructed image.
        criterion (torch.nn.Module): Criterion for calculating the loss.

    Returns:
        torch.Tensor: Cycle consistency loss.
    """
    return criterion(img, rec_img)

