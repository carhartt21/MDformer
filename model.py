import os
from os.path import join as ospj
import time
import datetime
from munch import Munch
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from initialize import build_model, set_criterions
from util.checkpoint import CheckpointIO
from data_loader import TrainProvider, TestProvider
import utils as utils
from metrics.eval import calculate_metrics
import loss
from PIL import Image
from typing import Dict, Tuple, Optional, List, Union

from torchvision.utils import draw_bounding_boxes


class StarFormer(nn.Module):
    def __init__(self, cfg, mode, device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model, self.model_ema = build_model(cfg)
        self.criterions = set_criterions(cfg=cfg, device=device)

        # below setattrs are to make networks be children of Solver, e.g., for self.to(self.device)
        logging.info("===== Networks architecture =====")
        for name, module in self.model.items():
            utils.print_network(module=module, name=name)
            setattr(self, name, module)
        for name, module in self.model_ema.items():
            setattr(self, name + "_ema", module)

        if mode == "train":
            self.optimizer = Munch()
            for net in self.model.keys():
                skip = {}
                skip_keywords = {}
                if hasattr(net, 'no_weight_decay'):
                    skip = net.no_weight_decay()
                if hasattr(net, 'no_weight_decay_keywords'):
                    skip_keywords = net.no_weight_decay_keywords()
                if "MLPHead" in net:
                    continue
                if net == "MappingNetwork" and cfg.TRAIN.lr_MN > 0.0:
                    lr = cfg.TRAIN.lr_MN
                elif net == "TransformerGen" and cfg.TRAIN.lr_G > 0.0:
                    lr = cfg.TRAIN.lr_G
                elif net == "Discriminator" and cfg.TRAIN.lr_D > 0.0:
                    lr = cfg.TRAIN.lr_D
                elif net == "ContentEncoder" and cfg.TRAIN.lr_CE > 0.0:
                    lr = cfg.TRAIN.lr_CE
                else:
                    lr = cfg.TRAIN.lr
                parameters = self._set_weight_decay(self.model[net], skip, skip_keywords)                    
                self.optimizer[net] = torch.optim.Adam(
                    params=parameters,
                    lr=lr,
                    betas=cfg.TRAIN.optim_beta,
                    weight_decay=cfg.TRAIN.weight_decay,
                )
            self.ckptios = [
                CheckpointIO(
                    ospj(cfg.DIR, cfg.MODEL.name, "ep_{}_model.ckpt"),
                    data_parallel=True,
                    **self.model,
                ),
                CheckpointIO(
                    ospj(cfg.DIR, cfg.MODEL.name, "ep_{}_model_ema.ckpt"),
                    data_parallel=True,
                    **self.model_ema,
                ),
                CheckpointIO(
                    ospj(cfg.DIR, cfg.MODEL.name, "ep_{}_optimizer.ckpt"),
                    **self.optimizer,
                ),
            ]
        else:
            self.ckptios = [
                CheckpointIO(
                    ospj(cfg.DIR, cfg.MODEL.name, "ep_{}_model_ema.ckpt"),
                    data_parallel=True,
                    **self.model_ema,
                )
            ]

        self.to(self.device)
        logging.info(f"===== Initialization =====")
        for name, network in self.named_children():
            if ("ema" not in name) and ("MLPHead" not in name):
                logging.info(f">> Initializing {name}")
                network.apply(utils.he_init)

    def _save_checkpoint(self, epoch):
        logging.info(f"===== Save intermediate weights =====")
        for ckptio in self.ckptios:
            ckptio.save(epoch)

    def _load_checkpoint(self, epoch):
        logging.info(f"===== Load pretrained weights =====")
        for ckptio in self.ckptios:
            ckptio.load(epoch)

    def _reset_grad(self):
        for optim in self.optimizer.values():
            optim.zero_grad()
            
    def _set_weight_decay(self, model, skip_list=(), skip_keywords=()):
        has_decay = []
        no_decay = []
        for name, param in model.module.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or _check_keywords(name, skip_keywords):
                no_decay.append(param)
                # print(f"{name} has no weight decay")
            else:
                has_decay.append(param)
        return [{'params': has_decay},
                {'params': no_decay, 'weight_decay': 0.}]        

          

    def train(self, loader, visualizer):
        cfg = self.cfg
        device = self.device
        model = self.model
        model_ema = self.model_ema
        optimizer = self.optimizer
        criterions = self.criterions

        for m in model.values():
            m.train()

        # fetch random validation images for debugging
        input_provider = TrainProvider(loader, cfg.MODEL.latent_dim, "train")
        # fetcher_val = InputFetcher(loaders.val, None, cfg.latent_dim, 'val')
        # inputs_val = next(fetcher_val)

        # resume training if necessary
        if cfg.TRAIN.start_epoch > 0:
            self._load_checkpoint(cfg.TRAIN.start_epoch)

        # remember the initial value of ds weight
        initial_lambda_ds = cfg.TRAIN.lambda_StyleDiv

        box_feature = torch.empty(1).to(device)

        logging.info("===== Start Training =====")
        logging.info(
            ">> Training for {} epoches with {} iterations per epoch".format(
                cfg.TRAIN.end_epoch - cfg.TRAIN.start_epoch,
                cfg.TRAIN.epoch_iters // cfg.TRAIN.batch_size_per_gpu,
            )
        )

        for epoch in range(cfg.TRAIN.start_epoch, cfg.TRAIN.end_epoch):
            visualizer.reset()
            iter_date_time = time.time()

            logging.info("===== Training progress: Epoch {} =====".format(epoch + 1))

            box_feature = torch.empty(1).to(device)
            for i in range(0, cfg.TRAIN.epoch_iters // cfg.TRAIN.batch_size_per_gpu):
                inputs, refs = next(input_provider)

                # recon_img = torch.zeros(fake_img.shape).to(device)

                # initialize MLPs the first iteration
                if (
                    epoch == cfg.TRAIN.start_epoch
                    and i == 0
                    and (
                        cfg.TRAIN.w_NCE != 0.0
                        or (
                            cfg.TRAIN.w_Instance_NCE != 0.0
                            and cfg.TRAIN.data.n_bbox > 0
                        )
                    )
                ):
                    parameter_F = []
                    _, _, features, _ = self.model_generation(
                        model=model,
                        inputs=inputs,
                        refs=refs,
                        lat_trg=inputs.lat_trg,
                        n_bbox=cfg.TRAIN.n_bbox,
                        from_lat=True,
                        feat_layers=cfg.MODEL.feat_layers,
                    )

                    if cfg.TRAIN.n_bbox > 0 and len(features) > len(
                        cfg.MODEL.feat_layers
                    ):
                        features, box_feature = features[:-1], features[-1]
                    if cfg.TRAIN.w_NCE != 0.0:
                        model.MLPHead.module.create_mlp(feats=features, device=device)
                    if cfg.TRAIN.w_Instance_NCE != 0.0 and cfg.TRAIN.n_bbox > 0:
                        model.MLPHeadInst.module.create_mlp([box_feature], device)
                    for key, val in model.items():
                        if "MLPHead" in key:
                            logging.info(f">> Initializing {key}")
                            # model[key] = nn.DataParallel(val)
                            parameter_F += list(val.parameters())
                    optimizer.MLPHead = torch.optim.Adam(
                        parameter_F, lr=float(cfg.TRAIN.lr)
                    )


                # generate image from latent vector
                fake_img_lat, _, features, s_trg = self.model_generation(
                    model=model,
                    inputs=inputs,
                    refs=refs,
                    lat_trg=inputs.lat_trg,
                    n_bbox=cfg.TRAIN.n_bbox,
                    from_lat=True,
                    feat_layers=cfg.MODEL.feat_layers,
                )
                # ----------------------------------------------------------
                # train the discriminator
                # ----------------------------------------------------------                

                total_D_loss, D_losses_lat = loss.compute_D_loss(
                    inputs=inputs,
                    refs=refs,
                    model=model,
                    fake_img=fake_img_lat,
                    criterions=criterions,
                    cfg=cfg,
                )
                # utils.set_requires_grad(model.Discriminator.module, True)
                self._reset_grad()
                total_D_loss.backward()
                optimizer.Discriminator.step()

                G_loss, G_losses_lat = loss.compute_G_loss(
                    model=model,
                    inputs=inputs,
                    fake_img=fake_img_lat,
                    features=features,
                    s_trg=s_trg,
                    refs=refs,
                    criterions=criterions,
                    cfg=cfg,
                )

                if cfg.TRAIN.w_Cycle > 0.0:
                    recon_img = self.model_reconstruction(
                        inputs=inputs,
                        fake_img=fake_img_lat,
                        model=model,
                        d_trg=refs.d_trg,
                    )
                    cycle_loss = cfg.TRAIN.w_Cycle * loss.compute_cycle_loss(
                        inputs.img_src, recon_img, criterions.Cycle
                    )
                    G_loss += cycle_loss
                    G_losses_lat.cycle_loss = cycle_loss
                # utils.set_requires_grad(model.Discriminator.module, False)
                self._reset_grad()
                G_loss.backward()

                optimizer.ContentEncoder.step()
                optimizer.TransformerGen.step()
                optimizer.TransformerEnc.step()
                optimizer.MLPAdain.step()
                optimizer.MappingNetwork.step()
                optimizer.StyleEncoder.step()
                if cfg.TRAIN.w_NCE > 0.0:
                    optimizer.MLPHead.step()

                # ---------------------------------------------------------
                # generate image from reference image
                # ---------------------------------------------------------
                fake_img_ref, fake_box, features, s_trg = self.model_generation(
                    model=model,
                    inputs=inputs,
                    refs=refs,
                    lat_trg=inputs.lat_trg,
                    n_bbox=cfg.TRAIN.n_bbox,
                    from_lat=False,
                    feat_layers=cfg.MODEL.feat_layers,
                )        


                #---------------------------------------------------------
                # train the discriminator            
                #---------------------------------------------------------
                total_D_loss, D_losses_ref = loss.compute_D_loss(
                    inputs=inputs,
                    refs=refs,
                    fake_img=fake_img_ref,
                    model=model,
                    criterions=criterions,
                    cfg=cfg,
                )

                self._reset_grad()
                total_D_loss.backward()
                optimizer.Discriminator.step()
                # ---------------------------------------------------------
                # train the generator
                # ---------------------------------------------------------
                G_loss, G_losses_ref = loss.compute_G_loss(
                    model=model,
                    fake_img=fake_img_ref,
                    s_trg=s_trg,
                    features=features,
                    inputs=inputs,
                    refs=refs,
                    criterions=criterions,
                    cfg=cfg,
                )

                if cfg.TRAIN.w_Cycle > 0.0:
                    recon_img = self.model_reconstruction(
                        inputs=inputs,
                        fake_img=fake_img_ref,
                        model=model,
                        d_trg=refs.d_trg,
                    )
                    cycle_loss = cfg.TRAIN.w_Cycle * loss.compute_cycle_loss(
                        inputs.img_src, recon_img, criterions.Cycle
                    )
                    G_loss += cycle_loss
                    G_losses_ref.cycle_loss = cycle_loss

                # utils.set_requires_grad(model.Discriminator.module, False)
                self._reset_grad()
                G_loss.backward()

                optimizer.ContentEncoder.step()
                optimizer.TransformerGen.step()
                optimizer.TransformerEnc.step()
                optimizer.MLPAdain.step()
                optimizer.MappingNetwork.step()
                optimizer.StyleEncoder.step()
                if cfg.TRAIN.w_NCE > 0.0:
                    optimizer.MLPHead.step()


                # compute moving average of network parameters
                moving_average(
                    model.MappingNetwork, model_ema.MappingNetwork, beta=0.999
                )
                moving_average(model.StyleEncoder, model_ema.StyleEncoder, beta=0.999)

                losses = Munch()
                losses.D_lat = D_losses_lat
                losses.G_lat = G_losses_lat
                losses.D_ref = D_losses_ref
                losses.G_ref = G_losses_ref

                # decay weight for diversity sensitive loss
                if cfg.TRAIN.lambda_StyleDiv > 0:
                    cfg.TRAIN.lambda_StyleDiv -= (
                        initial_lambda_ds / cfg.TRAIN.w_StyleDiv_iter
                    )

                total_iters = epoch * cfg.TRAIN.epoch_iters + i

                # print info to visdom
                if cfg.VISUAL.visdom:
                    if (i % cfg.VISUAL.display_losses_iter) == 0:
                        visualizer.plot_current_losses(
                            epoch, float(i) / len(loader), losses=losses
                        )
                    if (i % cfg.VISUAL.display_sample_iter) == 0:
                        current_visuals = {
                            "input_img": inputs.img_src,
                            "generated_img_lat": fake_img_lat,
                            "reference_img": refs.img_ref,
                            "generated_img_ref": fake_img_ref,
                        }
                        current_domains = {
                            "source_domain": inputs.d_src,
                            "target_domain": refs.d_trg,
                        }
                        visualizer.display_current_samples(
                            current_visuals,
                            current_domains,
                            epoch,
                            (total_iters % cfg.VISUAL.image_save_iter == 0),
                        )
                # print info to console
                if (i % cfg.VISUAL.print_losses_iter) == 0:
                    visualizer.print_current_losses(
                        epoch + 1, i, losses, time.time() - iter_date_time
                    )
                # save intermediate results
                if (
                    cfg.VISUAL.save_intermediate
                    and i % cfg.VISUAL.save_results_freq == 0
                ):
                    logging.info(
                        ">>>> Saving intermediate results to {}/{}".format(
                            cfg.TRAIN.log_path, cfg.MODEL.name
                        )
                    )
                    domain_imgs = utils.domain_to_image_tensor(
                        refs.d_trg, cfg.DATASET.target_domain_names, cfg.MODEL.img_size
                    )
                    output = torch.stack(
                        [
                            inputs.img_src,
                            domain_imgs,
                            fake_img_lat,
                            refs.img_ref,
                            fake_img_ref,
                        ],
                        dim=0,
                    ).flatten(0, 1)
                    utils.save_image_from_tensor(
                        output,
                        filename="{}/{}/{}_results.jpg".format(
                            cfg.TRAIN.log_path, cfg.MODEL.name, str(total_iters)
                        ),
                        normalize=cfg.TRAIN.img_norm,
                    )
                    if cfg.TRAIN.w_Instance_NCE > 0.0 and cfg.TRAIN.n_bbox > 0:
                        bbox = (inputs.bbox[0, :, 1:].cpu() * cfg.MODEL.img_size[0]).to(
                            torch.int
                        )
                        img = utils.denormalize(
                            inputs.img_src[0].unsqueeze(dim=0).cpu()
                        )
                        img = (
                            img.squeeze()
                            .mul(255)
                            .add_(0.5)
                            .clamp_(0, 255)
                            .to(torch.uint8)
                        )
                        img = draw_bounding_boxes(img, bbox).permute(1, 2, 0).numpy()
                        img = Image.fromarray(img)
                        img.save(
                            "{}/{}/source_image_with_bb_{}.jpg".format(
                                cfg.TRAIN.log_path, cfg.MODEL.name, str(total_iters)
                            )
                        )
            # Save model & optimizer and example images
            if epoch > 0 and (epoch % cfg.TRAIN.save_epoch) == 0:
                self._save_checkpoint(epoch=epoch)

                # generate images for debugging
                # if (i+1) % args.sample_every == 0:
                #     os.makedirs(args.sample_dir, exist_ok=True)
                #     utils.debug_image(model_ema, args, inputs=inputs_val, step=i+1)

                # # save model checkpoints
                # if (i+1) % args.save_every == 0:
                #     self._save_checkpoint(step=i+1)

                # compute FID and LPIPS if necessary
                # if (i+1) % args.eval_every == 0:
                #     calculate_metrics(model_ema, args, i+1, mode='latent')
                #     calculate_metrics(model_ema, args, i+1, mode='reference')

    @torch.no_grad()
    def sample(self, loaders):
        cfg = self.cfg
        model_ema = self.model_ema
        os.makedirs(cfg.result_dir, exist_ok=True)
        self._load_checkpoint(cfg.TEST.load_epoch)

        src = next(TestProvider(loaders.src, None, cfg.MODEL.latent_dim, "test"))
        ref = next(TestProvider(loaders.ref, None, cfg.MODEL.latent_dim, "test"))

        fname = ospj(cfg.TEST.result_dir, "reference.jpg")
        print("Working on {}...".format(fname))
        utils.translate_using_reference(model_ema, cfg, src.x, ref.x, ref.y, fname)

        # fname = ospj(args.result_dir, 'video_ref.mp4')
        # print('Working on {}...'.format(fname))
        # utils.video_ref(model_ema, args, src.x, ref.x, ref.y, fname)

    @torch.no_grad()
    def evaluate(self):
        args = self.args
        model_ema = self.model_ema
        resume_iter = args.resume_iter
        self._load_checkpoint(args.resume_iter)
        calculate_metrics(model_ema, args, step=resume_iter, mode="latent")
        calculate_metrics(model_ema, args, step=resume_iter, mode="reference")

    def model_generation(
        self,
        inputs: Union[torch.Tensor, dict],
        refs,
        lat_trg,
        model: dict,
        from_lat: bool = True,
        n_bbox: int = -1,
        feat_layers: List[str] = [],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        style_code_src = model.StyleEncoder(inputs.img_src, inputs.d_src)
        if from_lat:
            style_code_trg = model.MappingNetwork(lat_trg, refs.d_trg)
        else:
            style_code_trg = model.StyleEncoder(refs.img_ref, refs.d_trg)
        utils.apply_adain_params(
            model=model.TransformerEnc.module.transformer,
            adain_params=model.MLPAdain(style_code_src),
        )
        if "bbox" in inputs and n_bbox != -1:
            features += [
                model.TransformerEnc.module.embedding.extract_box_feature(
                    feat_content, inputs.bbox, n_bbox
                )
            ]

        if n_bbox == -1:
            aggregated_feat, _, _ = model.TransformerEnc(
                feat_content, sem_embed=True, sem_labels=inputs.seg, n_bbox=n_bbox
            )
        else:
            aggregated_feat, aggregated_box, weights = model.TransformerEnc(
                feat_content, sem_labels=inputs.seg, bbox_info=inputs.bbox, n_bbox=n_bbox
            )
        utils.apply_adain_params(
            model=model.TransformerGen.module, adain_params=model.MLPAdain(style_code_trg)
        )
        fake = model.TransformerGen(aggregated_feat)
        fake_box = (
            model.TransformerGen(aggregated_box, inputs.bbox)
            if n_bbox != -1 in inputs
            else None
        )
        # logging.info('model forward generation d_src_pred: {}'.format(d_src_pred))
        return fake, fake_box, features, style_code_trg


    def model_reconstruction(
        self,
        inputs: Union[torch.Tensor, dict],
        fake_img: torch.Tensor,
        model: dict,
        d_trg: torch.Tensor,
        feat_layers: List[str] = [],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        style_code_fake = model.StyleEncoder(fake_img, d_trg)
        style_code_rec = model.StyleEncoder(inputs.img_src, inputs.d_src)
        # use mapping network to generate a style code for the reconstruction
        utils.apply_adain_params(
            model.MLPAdain(style_code_fake),
            model.TransformerEnc.module.transformer,
        )
        # use segmenation map from source also for reconstruction
        aggregated_feat, _, _ = model.TransformerEnc(
            feat_content, sem_embed=True, sem_labels=inputs.seg, n_bbox=-1
        )
        utils.apply_adain_params(
            model=model.TransformerGen.module, adain_params=model.MLPAdain(style_code_rec)
        )
        # aggregated_feat, _, weights = model.TransformerEnc(feat_content, sem_embed=False, n_bbox=-1)
        rec_img = model.TransformerGen(aggregated_feat)
        return rec_img



def moving_average(model, model_test, beta=0.999):
    for param, param_test in zip(model.parameters(), model_test.parameters()):
        param_test.data = torch.lerp(param.data, param_test.data, beta)

def _check_keywords(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin  