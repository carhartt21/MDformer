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



######################################### Discriminator Loss #########################################


def compute_D_loss(
    inputs, refs, fake_img, model, criterions, cfg, from_lat=True, gan_mode="lsgan"
):
    total_D_loss = 0
    D_losses = Munch()
    D_losses.D_fake_loss = compute_Discrim_loss(
        fake_img,
        utils.batch_to_onehot(refs.d_trg),
        model.Discriminator,
        criterions.GAN,
        False,
    )
    if inputs.d_src.sum() > 0.0:
        D_losses.D_real_loss = compute_Discrim_loss(
            inputs.img_src, inputs.d_src, model.Discriminator, criterions.GAN, True
        )
    else:
        D_losses.D_real_loss = torch.tensor(0.0).to(inputs.img_src.device)
    # use the ref image and randomly change the domain to be eihter real of fake:
    if torch.rand(1) > 0.5:
        d_fake = utils.random_change_matrix(refs.d_trg)
        D_losses.D_loss_aux = compute_Discrim_loss(
            refs.img_ref, d_fake, model.Discriminator, criterions.GAN, False
        )
    else:
        D_losses.D_loss_aux = compute_Discrim_loss(
            refs.img_ref, refs.d_trg, model.Discriminator, criterions.GAN, True
        )

    if gan_mode == "vanilla":
        D_losses.D_reg_loss = cfg.TRAIN.w_l_reg * r1_reg(
            model.Discriminator, inputs.img_src, inputs.d_src
        )

    for _, loss in D_losses.items():
        total_D_loss += loss

    return total_D_loss / len(D_losses), D_losses


def compute_G_loss(
    inputs: Dict,
    refs: Dict,
    fake_img: torch.Tensor,
    s_trg: torch.Tensor,
    model: Dict,
    criterions: Dict,
    features: List[torch.Tensor],
    cfg: object,
    num_patches: int = 256,
) -> Tuple[float, Dict]:
    # fake_img, box_feature, features, s_trg = model_generation(
    #     inputs=inputs,
    #     refs=refs,
    #     lat_trg=inputs.lat_trg,
    #     model=model,
    #     n_bbox=cfg.TRAIN.n_bbox,
    #     feat_layers=cfg.MODEL.feat_layers,
    #     from_lat=from_lat,
    # )

    if cfg.TRAIN.n_bbox > 0 and len(features) > len(cfg.MODEL.feat_layers):
        features, box_feature = features[:-1], features[-1]

    G_losses = Munch()
    total_G_loss = 0
    # logging.info("Generating GAN loss...")
    # fake_img = fake_imgs[0]
    if cfg.TRAIN.w_GAN > 0.0:
        G_losses.GAN_loss = cfg.TRAIN.w_GAN * compute_GAN_loss(
            fake_img,
            utils.batch_to_onehot(refs.d_trg),
            model.Discriminator,
            criterions.GAN,
        )
    # logging.info("Generating Cycle loss...")
    if cfg.TRAIN.w_StyleRecon > 0.0:
        s_fake = model.StyleEncoder(fake_img, refs.d_trg)
        G_losses.style_loss = cfg.TRAIN.w_StyleRecon * compute_style_recon_loss(
            s_fake, s_trg, criterions.Idt
        )
    # logging.info("Generating NCE loss...")
    if cfg.TRAIN.w_NCE > 0.0 or (
        cfg.TRAIN.w_Instance_NCE > 0.0 and cfg.TRAIN.n_bbox > 0
    ):
        fake_feat_content, fake_features = model.ContentEncoder(
            fake_img, cfg.MODEL.feat_layers
        )
        if cfg.TRAIN.w_Instance_NCE > 0.0 and cfg.TRAIN.n_bbox > 0:
            fake_box_feature = (
                model.TransformerEnc.module.embedding.extract_box_feature(
                    fake_feat_content, inputs.bbox, cfg.TRAIN.n_bbox
                )
            )

    if cfg.TRAIN.w_NCE > 0.0:
        G_losses.NCE_loss = cfg.TRAIN.w_NCE * compute_NCE_loss(
            feat_q=features,
            feat_k=features,
            model=model.MLPHead,
            criterionNCE=criterions.NCE,
            num_patches=num_patches,
        )

    # if cfg.TRAIN.w_NCE > 0.0:
    #     G_losses.NCE_loss = cfg.TRAIN.w_NCE * compute_SemNCE_loss(
    #         feat_q=fake_features,
    #         feat_k=features,
    #         model=model.MLPHead2,
    #         criterionNCE=criterions.SemNCE,
    #         num_patches=cfg.MODEL.num_patches,
    #         seg=inputs.seg
    #     )

    if cfg.TRAIN.w_Instance_NCE > 0.0 and cfg.TRAIN.n_bbox > 0:
        valid_box = torch.where(inputs.bbox[:, :, 0] > 0, True, False).view(-1)
        if valid_box[valid_box == True].shape[0] == 0.0:
            G_losses.instNCE_loss = torch.tensor(0.0).to(inputs.img_src.device)
        else:
            criterions.InstNCE.batch_size = valid_box[valid_box == True].shape[0]
            G_losses.instNCE_loss = cfg.TRAIN.w_Instance_NCE * compute_InstNCE_loss(
                [fake_box_feature[valid_box, :, :, :]],
                [box_feature[valid_box, :, :, :]],
                model.MLPHeadInst,
                criterions.InstNCE,
                64,
            )

    # if cfg.TRAIN.w_StyleDiv > 0.0 and from_lat:
    #     fake_img_2, _, _, _ = model_generation(
    #         inputs=inputs,
    #         refs=refs,
    #         lat_trg=inputs.lat_trg_2,
    #         model=model,
    #         n_bbox=cfg.TRAIN.n_bbox,
    #         feat_layers=cfg.MODEL.feat_layers,
    #     )
    #     G_losses.style_div_loss = cfg.TRAIN.w_StyleDiv * compute_diversity_loss(
    #         fake_img, fake_img_2, criterions.Style_Div
    #     )


    # if cfg.TRAIN.w_DClass > 0.0:
    #     if d_src_pred is not None and torch.max(inputs.d_src) > 0.0:
    #         G_losses.class_loss = cfg.TRAIN.w_DClass * compute_domain_classification_loss(refs.d_trg, d_src_pred, criterions.DClass)
    #     else:
    #         G_losses.class_loss = torch.tensor(0.0).to(inputs.img_src.device)

    for key, loss in G_losses.items():
        if key == "style_div_loss":
            total_G_loss -= loss
        else:
            total_G_loss += loss

    return total_G_loss, G_losses


######################################### Each Loss #########################################


def compute_Discrim_loss(img, domain, model, criterion, Target=True):
    """
    Calculate the Discriminator loss.

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


def compute_style_ref_loss(src, tgt, criterion):
    """
    Calculate the style reference loss.

    Args:
        src (torch.Tensor): Source image.
        tgt (torch.Tensor): Target image.
        criterion (torch.nn.Module): Criterion for calculating the loss.

    Returns:
        torch.Tensor: Style reconstruction loss.
    """
    return criterion(src, tgt)


def compute_SemNCE_loss(feat_q, feat_k, model, criterionNCE, num_patches, seg=None):
    """
    Calculate the NCE (Normalized Cross Entropy) loss.

    Args:
        feat_q (List[torch.Tensor]): Query features from input image.
        feat_k (List[torch.Tensor]): Key features from fake image.
        model (torch.nn.Module): MLP model.
        criterionNCE (torch.nn.Module): Criterion for calculating the loss.
        num_patches (int): Number of patches.

    Returns:
        torch.Tensor: NCE loss.
    """

    # sample from fake image
    feat_q_pool, sample_ids, _, _ = model(
        feats=feat_q, num_patches=num_patches, patch_ids=None
    )
    # sample from real image
    feat_k_pool, _, feat_k_pos, feat_k_neg = model(
        feats=feat_k, num_patches=num_patches, patch_ids=sample_ids, seg=seg
    )

    total_nce_loss = 0.0
    for f_q, f_k, f_k_p, f_k_n in zip(feat_q_pool, feat_k_pool, feat_k_pos, feat_k_neg):
        total_nce_loss += criterionNCE(f_q, f_k, f_k_p, f_k_n)

    return total_nce_loss


def compute_NCE_loss(feat_q, feat_k, model, criterionNCE, num_patches, seg=None):
    """
    Calculate the NCE (Normalized Cross Entropy) loss.

    Args:
        feat_q (List[torch.Tensor]): Query features from input image.
        feat_k (List[torch.Tensor]): Key features from fake image.
        model (torch.nn.Module): MLP model.
        criterionNCE (torch.nn.Module): Criterion for calculating the loss.
        num_patches (int): Number of patches.

    Returns:
        torch.Tensor: NCE loss.
    """

    feat_k_pool, sample_ids = model(
        feats=feat_k, num_patches=num_patches, patch_ids=None
    )
    feat_q_pool, _ = model(feats=feat_q, num_patches=num_patches, patch_ids=sample_ids)

    total_nce_loss = 0.0
    for f_q, f_k in zip(feat_q_pool, feat_k_pool):
        total_nce_loss += criterionNCE(f_q, f_k).mean()

    return total_nce_loss / len(feat_q)


def compute_InstNCE_loss(feat_q, feat_k, model, criterionNCE, num_patches):
    """
    Calculate the NCE (Normalized Cross Entropy) loss.

    Args:
        feat_q (List[torch.Tensor]): Query features from input image.
        feat_k (List[torch.Tensor]): Key features from fake image.
        model (torch.nn.Module): MLP model.
        criterionNCE (torch.nn.Module): Criterion for calculating the loss.
        num_patches (int): Number of patches.

    Returns:
        torch.Tensor: NCE loss.
    """

    feat_k_pool, sample_ids = model(
        feats=feat_k, num_patches=num_patches, patch_ids=None
    )

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
    trg_idx[max == 0] = -1
    # logging.info('src_pred: {} trg_idx: {}'.format(d_src_pred, trg_idx))
    test = criterion(d_src_pred, trg_idx)
    # logging.info('Domain Classification Loss: {}'.format(test))
    # logging.info(f"trg_idx: {trg_idx} d_src_pred: {d_src_pred}")
    # logging.info(f"Domain Classification Loss: {test} {test.shape} {test.dtype}")
    return test


def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(),
        inputs=x_in,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert grad_dout2.size() == x_in.size()
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg


def adv_loss(logits, target):
    assert target in [1, 0]
    targets = torch.full_like(logits, fill_value=target)
    loss = torch.functional.binary_cross_entropy_with_logits(logits, targets)
    return loss
