import torch
import torchvision
import torch.nn as nn
import numpy as np
import logging
from munch import Munch

import utils
import pdb
from typing import Dict, Tuple, Optional, List, Union
import torch


######################################### Forward #########################################

def model_forward_generation(inputs: Union[torch.Tensor, dict], refs, lat_trg, model: dict, n_bbox: int = -1, feat_layers: List[str] = []) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Forward pass of the generation model.

    Args:
        inputs (Union[torch.Tensor, dict]): Input tensor or dictionary containing input tensors.
        model (dict): Dictionary containing the model components.
        n_bbox (int, optional): Number of bounding boxes. Defaults to -1.
        feat_layers (List[str], optional): List of feature layers. Defaults to [].

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple containing the generated fake image, fake box, and features.
    """
    # when performing generation, the input is the content image from the data loader
    feat_content, features = model.ContentEncoder(inputs.img_src, feat_layers)
    style_code = model.MappingNetwork(lat_trg, refs.d_trg)
    utils.assign_adain_params(model.MLP_Adain(style_code), model.Transformer.module.transformer.layers)
    if 'bbox' in inputs and n_bbox != -1:
        features += [model.Transformer.module.extract_box_feature(feat_content, inputs.bbox, n_bbox)]

    if n_bbox == -1:
        aggregated_feat, _, _ = model.Transformer(feat_content, sem_embed=True, sem_labels=inputs.seg, n_bbox=n_bbox)
    else:
        aggregated_feat, aggregated_box, weights = model.Transformer(feat_content, sem_labels=inputs.seg, bbox_info=inputs.bbox, n_bbox=n_bbox)        

    d_src_pred = model.DomainClassifier(aggregated_feat)
    fake = model.Generator(aggregated_feat)
    fake_box = model.Generator(aggregated_box, inputs.bbox) if n_bbox != -1 in inputs else None
    # logging.info('model forward generation d_src_pred: {}'.format(d_src_pred))
    return fake, fake_box, features, d_src_pred


def model_forward_reconstruction(inputs: Union[torch.Tensor, dict], fake_img: torch.Tensor, model: dict, d_src_pred: torch.Tensor, feat_layers: List[str] = []) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Forward pass of the model for image reconstruction.

    Args:
        fake_img (torch.Tensor): The input fake image tensor.
        model (dict): The model dictionary containing the content encoder, style encoder, MLP Adain, transformer, and generator.
        d_src_pred (torch.Tensor): The source domain tensor.
        feat_layers (List[str], optional): List of feature layers. Defaults to [].

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the reconstructed image tensor, style code tensor, and aggregated feature tensor.
    """
    feat_content, _ = model.ContentEncoder(fake_img, feat_layers)

    # replace empty domain with predicted domain
    d_recon_trg = inputs.d_src.clone()
    max_values, _ = torch.max(d_recon_trg, dim=1)
    d_recon_trg[max_values == 0] = d_src_pred[max_values == 0].to(d_recon_trg.dtype)
    # style_code = model.StyleEncoder(inputs.img_src, torch.argmax(d_recon_trg, dim=1))
    lat_recon_trg = torch.randn(inputs.lat_trg.shape).to(inputs.lat_trg.device)
    # use mapping network to generate a style code for the reconstruction
    recon_style_code = model.MappingNetwork(lat_recon_trg, d_recon_trg)
    utils.assign_adain_params(model.MLP_Adain(recon_style_code), model.Transformer.module.transformer.layers)
    # use segmenation map from source also for reconstruction
    aggregated_feat, _, _ = model.Transformer(feat_content, sem_embed=True, sem_labels=inputs.seg, n_bbox=-1)
    # aggregated_feat, _, weights = model.Transformer(feat_content, sem_embed=False, n_bbox=-1)
    d_fake_img_pred = model.DomainClassifier(aggregated_feat)
    rec_img = model.Generator(aggregated_feat)
    return rec_img, recon_style_code, d_fake_img_pred


######################################### Total Loss #########################################

def compute_D_loss(inputs, refs, fake_img, model_D, criterions):
    """
    Calculate GAN loss for the discriminator.

    Args:
        inputs (Tensor): The input tensor.
        fake_img (Tensor): The fake image tensor.
        model_D (dict): The dictionary containing the discriminator model.
        criterions (dict): The dictionary containing the GAN loss criterions.

    Returns:
        tuple: A tuple containing the total discriminator loss and a dictionary of individual discriminator losses.
    """
    D_losses = Munch()  
    D_losses.D_fake_loss = compute_Discrim_loss(fake_img, utils.batch_to_onehot(refs.d_trg), model_D.Discrim, criterions.GAN, False) 
    if inputs.d_src.sum() > 0.0:
        D_losses.D_real_loss = compute_Discrim_loss(inputs.img_src, inputs.d_src, model_D.Discrim, criterions.GAN, True)
    else:
        D_losses.D_real_loss = torch.tensor(0.0).to(inputs.img_src.device)
    # use the ref image and randomly change the domain to be eihter real of fake:
    if torch.rand(1) > 0.5:
        d_fake = utils.random_change_matrix(refs.d_trg)
        D_losses.D_loss_aux = compute_Discrim_loss(refs.img_ref, d_fake, model_D.Discrim, criterions.GAN, False)
    else:
        D_losses.D_loss_aux = compute_Discrim_loss(refs.img_ref, refs.d_trg, model_D.Discrim, criterions.GAN, True)
    total_D_loss = (D_losses.D_fake_loss +  D_losses.D_real_loss + D_losses.D_loss_aux) / 3.0

    return total_D_loss, D_losses


def compute_G_loss(inputs: Dict, 
                   refs: Dict, 
                   fake_imgs: List[torch.Tensor], 
                   recon_img: torch.Tensor, 
                   features: torch.Tensor, 
                   box_feature: torch.Tensor, 
                   model_G: Dict, 
                   model_D: Dict, 
                   model_F: Dict, 
                   criterions: Dict, 
                   s_trg: torch.Tensor,
                   cfg: object) -> Tuple[float, Dict]:
    """
    Calculate loss for the generator.

    Args:
        inputs (Dict): Input data dictionary.
        fake_imgs (List[torch.Tensor]): Fake images generated by the generator.
        recon_img (torch.Tensor): Reconstructed image.
        style_code (torch.Tensor): Style code.
        features (torch.Tensor): Features.
        box_feature (torch.Tensor): Box feature.
        model_G (Dict): Generator model dictionary.
        model_D (Dict): Discriminator model dictionary.
        model_F (Dict): MLP model dictionary.
        criterions (Dict): Loss criterions dictionary.
        cfg (object): Configuration object.

    Returns:
        Tuple[float, Dict]: A tuple containing the total generator loss and a dictionary of individual losses.
    """
    G_losses = Munch()
    total_G_loss = 0

    fake_img = fake_imgs[0]
    if cfg.TRAIN.w_GAN > 0.0:
        G_losses.GAN_loss = cfg.TRAIN.w_GAN * compute_GAN_loss(fake_img, utils.batch_to_onehot(refs.d_trg), model_D.Discrim, criterions.GAN)
    
    if cfg.TRAIN.w_Recon > 0.0:
        s_fake = model_G.StyleEncoder(fake_img, torch.argmax(refs.d_trg, dim=1))
        G_losses.style_loss = cfg.TRAIN.w_Recon * compute_style_recon_loss(s_fake, s_trg, criterions.Idt)
    
    if cfg.TRAIN.w_NCE > 0.0 or (cfg.TRAIN.w_Instance_NCE > 0.0 and cfg.DATASET.n_bbox > 0):
        fake_feat_content, fake_features = model_G.ContentEncoder(fake_img, cfg.MODEL.feat_layers)
        if cfg.TRAIN.w_Instance_NCE > 0.0 and cfg.DATASET.n_bbox > 0:
            fake_box_feature = model_G.Transformer.module.extract_box_feature(fake_feat_content, inputs.bbox, cfg.DATASET.n_bbox)

    if cfg.TRAIN.w_NCE > 0.0:
        G_losses.NCE_loss = cfg.TRAIN.w_NCE * compute_NCE_loss(fake_features, features, model_F.MLP_head, criterions.NCE, cfg.MODEL.num_patches)
    if cfg.TRAIN.w_Instance_NCE > 0.0 and cfg.DATASET.n_bbox > 0:
        valid_box=torch.where(inputs.bbox[:,:,0] > 0, True,False).view(-1)
        if valid_box[valid_box ==  True].shape[0] == 0.0:
            G_losses.instNCE_loss = torch.tensor(0.0).to(inputs.img_src.device)
        else:
            criterions.InstNCE.batch_size = valid_box[valid_box ==  True].shape[0]
            G_losses.instNCE_loss = cfg.TRAIN.w_Instance_NCE * compute_NCE_loss([fake_box_feature[valid_box,:,:,:]], [box_feature[valid_box,:,:,:]], model_F.MLP_head_inst, criterions.InstNCE, 64)
    
    if len(fake_imgs) and cfg.TRAIN.w_StyleDiv > 0.0:
        G_losses.style_div_loss = cfg.TRAIN.w_StyleDiv * compute_diversity_loss(fake_img, fake_imgs[1], criterions.Style_Div)
    
    if cfg.TRAIN.w_Cycle > 0.0:
        G_losses.cycle_loss = cfg.TRAIN.w_Cycle * compute_cycle_loss(inputs.img_src, recon_img, criterions.Cycle)

    # if cfg.TRAIN.w_DClass > 0.0:
    #     if d_src_pred is not None and torch.max(inputs.d_src) > 0.0:
    #         G_losses.class_loss = cfg.TRAIN.w_DClass * compute_domain_classification_loss(refs.d_trg, d_src_pred, criterions.DClass)
    #     else: 
    #         G_losses.class_loss = torch.tensor(0.0).to(inputs.img_src.device)
            
    for key, loss in G_losses.items():
        if (key == 'style_div_loss'):
            total_G_loss -= loss
        else: 
            total_G_loss += loss

    return total_G_loss, G_losses


######################################### Each Loss #########################################



def compute_Discrim_loss(img, domain, model, criterion, Target=True):
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
    pred = model(img.detach(), domain)
    return criterion(pred, Target).mean()


def compute_GAN_loss(fake, target_domain, model, criterion):
    """
    Calculate the GAN loss.

    Args:
        fake (torch.Tensor): Fake image.
        model (torch.nn.Module): Discriminator model.
        criterion (torch.nn.Module): Criterion for calculating the loss.

    Returns:
        torch.Tensor: GAN loss.
    """
    pred_fake = model(fake, target_domain)
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
    feat_k_pool, sample_ids = model(feats=feat_k, num_patches=num_patches, patch_ids=None)
    feat_q_pool, _ = model(feats=feat_q, num_patches=num_patches, patch_ids=sample_ids)

    total_nce_loss = 0.0    
    for f_q, f_k in zip(feat_q_pool, feat_k_pool):
        total_nce_loss += criterionNCE(f_q, f_k).mean()

    return total_nce_loss / len(feat_q)


def compute_diversity_loss(img1, img2, criterion):
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

def compute_domain_classification_loss(d_src, d_src_pred, criterion):
    """
    Calculate the domain classification loss.

    Args:
        d_src (torch.Tensor): Source domain.
        d_src_pred (torch.Tensor): Predicted source domain.
        criterion (torch.nn.Module): Criterion for calculating the loss.

    Returns:
        torch.Tensor: Domain classification loss.
    """
    max, trg_idx = d_src.max(dim=1)
    # replace empty source domain with ignore index
    trg_idx[max==0] = -1
    # logging.info('src_pred: {} trg_idx: {}'.format(d_src_pred, trg_idx))
    test = criterion(d_src_pred, trg_idx)
    # logging.info('Domain Classification Loss: {}'.format(test))
    # logging.info(f"trg_idx: {trg_idx} d_src_pred: {d_src_pred}")
    # logging.info(f"Domain Classification Loss: {test} {test.shape} {test.dtype}")
    return test