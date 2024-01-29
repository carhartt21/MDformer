import os
from packaging import version
from PIL import Image, ImageDraw, ImageFont
import torchvision.utils as vutils

from argparse import Namespace
import cv2
import numpy as np
import torch
from torch import nn
import logging
import sys
import torch
from copy import deepcopy
import json
import random
import torch.nn.functional as F
from einops import repeat, rearrange


##################################### Visualize ##################################### 
def denormalize(x, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    # 3, H, W, B
    ten = x.clone().permute(1, 2, 3, 0)
    for t, m, s in zip(ten, mean, std):
        t.mul_(s).add_(m)
    # B, 3, H, W
    return torch.clamp(ten, 0, 1).permute(3, 0, 1, 2)

def domain_to_image_tensor(d_trg:torch.Tensor, target_domains:list, img_size=(320, 320)):
    """Convert one-hot encoded domains to image tensors"""
    domain_imgs = []
    domains = d_trg.argmax(dim=1)
    for i in range(d_trg.shape[0]):
        domain_imgs.append(text_to_img(target_domains[domains[i]], img_size))
    return torch.stack(domain_imgs, dim=0).to(d_trg.device)



def text_to_img(text:str, img_size=(320, 320)):
    """Convert domain name image as tensor"""
    # Create a white square image
    image = Image.new('RGB', img_size , 'white')

    # Draw image with text using PIL    
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=80)  # or specify a font with ImageFont.truetype()
    _ , _, text_width, text_height = draw.textbbox((0, 0), text, font=font)
    position = ((img_size[0] - text_width) // 2, (img_size[1] - text_height) // 2)
    draw.text(position, text, fill="black", font=font)

    # Convert the PIL Image to torch Tensor
    numpy_image = np.array(image)
    tensor_image = torch.from_numpy(numpy_image)
    tensor_image = tensor_image.permute(2, 0, 1)

    return tensor_image

    

@torch.no_grad()
def translate_using_latent(model_G, cfg, input, d_trg, psi, filename, latent_dim=64, feat_layers=[]):
    N, C, H, W = input.img.size()
    # latent_dim = z_trg_list[0].size(1)
    x_concat = [input.img]
    z_trg = torch.randn(cfg.TEST.num_images_per_domain, 1, cfg.MODEL.latent_dim).repeat(1, N, 1).to(input.img.device)

    z_many = torch.randn(10000, latent_dim).to(input.img.device)
    y_many = repeat(d_trg, 'b -> n b', n=10000).to(input.img.device)
    # y_many = torch.LongTensor(10000).to(input.img.device).fill_(d_trg)
    s_many = model_G.MappingNetwork(z_many, y_many)
    s_avg = torch.mean(s_many, dim=0, keepdim=True)
    s_avg = s_avg.repeat(N, 1)
    d_trg = rearrange(d_trg, 'b -> 1 b')
    s_trg = model_G.MappingNetwork(z_trg, d_trg)
    s_trg = torch.lerp(s_avg, s_trg, 0.0)
    assign_adain_params(model_G.MLPAdain(s_trg), model_G.Transformer.transformer.layers)
    feat_content, features = model_G.ContentEncoder(input.img, feat_layers)    
    aggregated_feat, _, _ = model_G.Transformer(feat_content, sem_embed=True, sem_labels=input.seg, n_bbox=-1)
     
    x_fake = model_G.Generator(aggregated_feat)    
    # x_concat += [x_fake]

    # x_concat = torch.cat(x_concat, dim=0)
    save_image_from_tensor(x_fake, N, filename)


def batch_to_onehot(matrix):
    """
    Translate every row of a matrix to a one-hot vector.

    Args:
        matrix (torch.Tensor): Input matrix.

    Returns:
        torch.Tensor: Matrix with one-hot vectors.
    """
    max_values, _ = torch.max(matrix, dim=1)
    one_hot = torch.eye(matrix.shape[1]).to(matrix.device)[torch.argmax(matrix, dim=1)]
    one_hot[max_values == 0] = 0
    return one_hot

def onehot_to_domain(onehot, target_domain_names=[], domain_dict='helper/data_cfg/domain_dict.json'):
    """
    Convert one-hot vector to domain name.

    Args:
        onehot (torch.Tensor): The one-hot vector.
        domain_dict (str): The path to the domain dictionary. Default is 'helper/data_cfg/domain_dict.json'.
    Returns:
        str: The domain name.
    """
    with open(domain_dict, 'r') as f:
        domain_idxs = json.load(f)
    idxs, n_dom = get_domain_indexes(target_domain_names)
    assert len(idxs) == onehot.shape[0], "The number of domains in the one-hot vector does not match the number of target domains."
    onehot_all = torch.zeros(n_dom, dtype=onehot.dtype, device=onehot.device)
    onehot_all[idxs] = onehot
    if onehot_all.sum() == 0:
        return 'unknown'
    domain_keys = [value for key, value in domain_idxs.items() if int(key) == onehot_all.argmax().item()]
    return domain_keys[0]

def domain_to_onehot(domain, target_domain_names):
    """Convert domain name to one-hot vector."""
    idx, n_dom = get_domain_indexes([domain])
    onehot = torch.zeros(n_dom, dtype=torch.int)
    onehot[idx] = 1    
    idxs, _ = get_domain_indexes(target_domain_names) 
    return onehot[idxs]

def random_change_domain(vector):
    """
    Randomly changes the 1 in a one-hot vector to a different index.

    Args:
        vector (torch.Tensor): The one-hot vector.

    Returns:
        torch.Tensor: The modified one-hot vector.
    """
    out_v = vector.clone()
    index_one = torch.nonzero(out_v)
    index_zero = torch.nonzero(out_v == 0)
    index_new_one = index_zero[torch.randperm(len(index_zero))[0]]
    out_v[index_one] = 0
    out_v[index_new_one] = 1
    return out_v

def random_change_matrix(matrix):
    """
    Randomly changes the 1 in each row of a matrix of one-hot vectors to a different index.

    Args:
        matrix (torch.Tensor): The matrix of one-hot vectors.

    Returns:
        torch.Tensor: The modified matrix.
    """
    modified_matrix = torch.zeros_like(matrix)
    for i in range(matrix.shape[0]):
        modified_matrix[i] = random_change_domain(matrix[i])
    return modified_matrix


def get_domain_indexes(target_domain_names, domain_dict='helper/data_cfg/domain_dict.json'):
    """
    Get the domain indexes.

    Args:
        target_domain_names (List[str]): The list of target domains.
        domain_dict (str): The path to the domain dictionary. Default is 'helper/data_cfg/domain_dict.json'.
    Returns:
        Dict[str, int]: A dictionary mapping each target domain to its index.
        int: The number of domains in the dictionary.
    """
    with open(domain_dict, 'r') as f:
        domain_idxs = json.load(f)
    domain_keys = [int(key) for key, value in domain_idxs.items() if value in target_domain_names]
    
    return domain_keys, len(domain_idxs)

def mask_to_bboxes(bin_mask, min_size = 512, pos_thres=0.5):
    """Cluster heatmap into discrete bounding boxes

    :param torch.Tensor[H, W] bin_mask: Binary mask
    :param float pos_thres: Threshold for assigning probability to positive class
    :param Optional[float] nms_thres: Threshold for non-max suppression (or ``None`` to skip)
    :param Optional[float] score_thres: Threshold for final bbox scores (or ``None`` to skip)
    :return Tuple[torch.Tensor]: Containing
        * bboxes[N, C=4]: bounding box coordinates in ltrb format
        * scores[N]: confidence scores (averaged across all pixels in the box)
    """

    def get_roi(data, bounds):
        """Extract region of interest from a tensor

        :param torch.Tensor[H, W] data: Original data
        :param dict bounds: With keys for left, right, top, and bottom
        :return torch.Tensor[H', W']: Subset of the original data
        """
        compound_slice = (
            slice(bounds['top'], bounds['bottom']),
            slice(bounds['left'], bounds['right']))
        return data[compound_slice]

    def is_covered(x, y, bbox):
        """Determine whether a point is covered/inside a bounding box

        :param int x: Point x-coordinate
        :param int y: Point y-coordinate
        :param torch.Tensor[int(4)] bbox: In ltrb format
        :return bool: Whether all boundaries are satisfied
        """
        left, top, right, bottom = bbox
        bounds = [
            x >= left,
            x <= right,
            y >= top,
            y <= bottom]
        return all(bounds)
    

    # Determine indices of each positive pixel
    # bin_mask = heatmap
    # bin_mask = torch.where(heatmap == 1, 1, 0)
    mask = torch.ones(bin_mask.size()).type_as(bin_mask)
    idxs = torch.flip(torch.nonzero(bin_mask*mask), [1])
    heatmap_height, heatmap_width = bin_mask.shape

    # Limit potential expansion to the heatmap boundaries
    edge_names = ['left', 'top', 'right', 'bottom']
    limits = {
        'left': 0,
        'top': 0,
        'right': heatmap_width,
        'bottom': heatmap_height}
    bboxes = []
    scores = []

    # Iterate over positive pixels
    for x, y in idxs:

        # Skip if an existing bbox already covers this point
        already_covered = False
        for bbox in bboxes:
            if is_covered(x, y, bbox):
                already_covered = True
                break
        if already_covered:
            continue

        incrementers = {k: 1 for k in edge_names}
        max_bounds = {
            'left': deepcopy(x),
            'top': deepcopy(y),
            'right': deepcopy(x),
            'bottom': deepcopy(y)}
        while True:

            # Extract the new, expanded ROI around the current (x, y) point
            bounds = {
                'left': max(limits['left'], x - incrementers['left']),
                'top': max(limits['top'], y - incrementers['top']),
                'right': min(limits['right'], x + incrementers['right'] + 1),
                'bottom': min(limits['bottom'], y + incrementers['bottom'] + 1)}
            roi = get_roi(bin_mask, bounds)

            # Get the vectors along each edge
            edges = {
                'left': roi[:, 0],
                'top': roi[0, :],
                'right': roi[:, -1],
                'bottom': roi[-1, :]}

            keep_going = False
            for k, v in edges.items():
                if v.sum()/v.numel() > pos_thres and limits[k] != max_bounds[k]:
                    keep_going = True
                    max_bounds[k] = bounds[k]
                    incrementers[k] += 1

            if not keep_going:
                final_roi = get_roi(bin_mask, max_bounds)
                if final_roi.numel() > min_size:
                    bboxes.append([max_bounds[k] - 1 if i > 1 else max_bounds[k] 
                                   for i, k in enumerate(edge_names)])
                break
    return torch.tensor(bboxes)

def save_component(log_path, model_name, epoch, model, optimizer, net_name='G'):
    """Save model and optimizer."""
    save_folder = os.path.join(log_path, model_name, "weights_{}".format(epoch+1))

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    #model_save
    for key, val in model.items():
        save_model_name = os.path.join(save_folder,"{}.pth".format(key))
        torch.save(val.state_dict(), save_model_name)

    #optimizer save
    save_optim_name = os.path.join(save_folder, "adam_{}.pth".format(net_name))
    torch.save(optimizer.state_dict(), save_optim_name)


def model_mode(model, mode = 0): 
    """Set model mode to train or eval."""
    for m in model.values():
        if mode == 0: #TRAIN
            m.train()
        else:
            m.eval()

def print_network(module, name):
    num_params = sum(p.numel() for p in module.parameters())
    logging.info(f'>> # Parameters {name}: {num_params:,}')


def he_init(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

def segmentation_to_bbox(mask, bg_classes = [0,1,2]):
    """Transfer a segmentation mask derived from an image file to a dictionary of bounding boxes for each class in the map.
    Remove backgound classes from a list of classes.
    Args:
        mask (np.array): A segmentation mask derived from an image file.
        bg_classes (list): A list of background classes to be removed from the mask.
        """
    # Get the unique classes in the mask
    classes = np.unique(mask)
    # Remove the background class
    classes = classes[classes != bg_classes]
    # Initialize the dictionary of bounding boxes
    bboxes = {}
    # Loop through the classes
    for cls in classes:
        # Get the coordinates of the pixels of the class
        coords = np.argwhere(mask == cls)
        # Get the minimum and maximum x and y coordinates
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)
        # Add the bounding box to the dictionary
        bboxes[cls] = [x_min, y_min, x_max, y_max]
    return bboxes

def setup_logger(distributed_rank=0, filename="log.txt"):
    logger = logging.getLogger("Logger")
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    fmt = "[%(asctime)s %(levelname)s %(filename)s] %(message)s"
    ch.setFormatter(logging.Formatter(fmt))
    logger.addHandler(ch)

    return logger    

def label_to_onehot(gt, num_classes, ignore_index=-1):
    '''
    Converts segmentation label to one hot format
    gt: ground truth label with size (N, H, W)
    num_classes: number of classes
    ignore_index: index(es) for ignored classes
    '''
    N, _, _ = gt.size()
    gt_ = gt
    gt_[gt_ == ignore_index] = num_classes
    onehot = torch.zeros(N, gt_.size(1), gt_.size(2), num_classes + 1)
    onehot = onehot.scatter_(-1, gt_.unsqueeze(-1), 1) 

    return onehot.permute(0, 3, 1, 2)

def onehot_to_class(onehot):
    batch_size = onehot.squeeze().shape[0]
    cls = torch.zeros(batch_size, dtype=torch.long)
    if len(onehot.shape) > 1:
        for i in range(onehot.shape[0]):
            _cls = torch.nonzero((onehot[i] + 1))
            if _cls.nelement() > 0:
                cls[i] = _cls
            else:
                cls[i] = torch.tensor([-1])
        return cls.cuda()
    else:
        _cls = torch.nonzero(onehot + 1)
        if _cls.nelement() > 0:
            cls = _cls
        else:
            cls = torch.tensor([-1])
        return cls.cuda()

def user_scattered_collate(batch):
    return batch

def assign_adain_params(adain_params, model):
    # assign the adain_params to the AdaIN layers in model
    for layer_id, layer in enumerate(model):
        m = layer[0].norm

        mean = adain_params[:, :m.num_features]
        std = adain_params[:, m.num_features:2*m.num_features]
        
        m.bias = mean.contiguous().view(-1)
        m.weight = std.contiguous().view(-1)
        if adain_params.size(1) > 2*m.num_features:
            adain_params = adain_params[:, 2*m.num_features:]
        n = layer[1].norm

        mean = adain_params[:, :n.num_features]
        std = adain_params[:, n.num_features:2*n.num_features]
        n.bias = mean.contiguous().view(-1)
        n.weight = std.contiguous().view(-1)
        if adain_params.size(1) > 2*n.num_features:
            adain_params = adain_params[:, 2*n.num_features:]

def get_num_adain_params(model):
    # return the number of AdaIN parameters needed by the model
    num_adain_params = 0
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
            num_adain_params += 2*m.num_features
    return num_adain_params            

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def save_color(tensor, path, name):
    color = np.split(tensor.clone().detach().permute(0,2,3,1).cpu().numpy(), tensor.shape[0], axis=0)
    for i, img in enumerate(color):
        img *= 255
        cv2.imwrite(f'{path}/{name}_{i}.png', cv2.cvtColor(np.squeeze(img.astype(np.uint8), axis=0), cv2.COLOR_BGR2RGB)) 

##################################### Visualize ##################################### 

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].clamp(-1.0, 1.0).cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    try:
        img = image_numpy.astype(imtype)
        return img
    except RuntimeError:
        return np.ones_like(image_numpy.transpose([2, 0, 1])) * 255

def save_image_from_tensor(x, filename, normalize='default'):
    _x = x.clone().detach()
    if normalize == 'imagenet':
        mean = torch.Tensor([0.485, 0.456, 0.406])
        std = torch.Tensor([0.229, 0.224, 0.225])
        _x = denormalize(_x, mean=mean, std=std)
    elif normalize == 'default':
        mean = torch.Tensor([0.5, 0.5, 0.5])
        std = torch.Tensor([0.5, 0.5, 0.5])
        _x = denormalize(_x, mean=mean, std=std)
    else: 
        _x = x
    # grid = vutils.make_grid(_x)
    # ndarr = grid.clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    vutils.save_image(_x.cpu(), filename, padding=0)

def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio is None:
        pass
    elif aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    elif aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

def copyconf(default_opt, **kwargs):
    conf = Namespace(**vars(default_opt))
    for key in kwargs:
        setattr(conf, key, kwargs[key])
    return conf

##################################### Visualize ##################################### 


##################################### Losses ##################################### 

class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode='lsgan', target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp', 'nonsaturating']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and ground truth labels.

        Parameters:
            prediction (tensor) - - typically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        bs = prediction.size(0)
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        elif self.gan_mode == 'nonsaturating':
            if target_is_real:
                loss = F.softplus(-prediction).view(bs, -1).mean(dim=1)
            else:
                loss = F.softplus(prediction).view(bs, -1).mean(dim=1)
        return loss
    

class PatchNCELoss(nn.Module):
    def __init__(self, batch_size, nce_T=0.07):
        """
        PatchNCELoss is a custom loss function for patch-based contrastive learning.

        Args:
            batch_size (int): The batch size of the input data.
            nce_T (float, optional): The temperature parameter for the loss calculation. Defaults to 0.07.
        """
        super().__init__()
        self.batch_size = batch_size
        self.nce_T = nce_T
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def forward(self, feat_q, feat_k):
        """
        Forward pass of the PatchNCELoss.

        Args:
            feat_q (torch.Tensor): The query features.
            feat_k (torch.Tensor): The key features.

        Returns:
            torch.Tensor: The computed loss.
        """
        batchSize = feat_q.shape[0] 
        dim = feat_q.shape[1] 
        feat_k = feat_k.detach()

        l_pos = torch.bmm(feat_q.view(batchSize, 1, -1), feat_k.view(batchSize, -1, 1))
        l_pos = l_pos.view(batchSize, 1)

        batch_dim_for_bmm = self.batch_size

        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1) 
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1)) 

        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        # # continue helper
        #  for i in range(batch_size):
        #     # Selecting negative patches within the same sample
        #     neg_indices = self.get_negative_indices(num_patches, i, batch_size)
        #     negatives.append(feat_k[neg_indices])
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.nce_T

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))

        return loss
    

class SemPatchNCELoss(nn.Module):
    def __init__(self, batch_size, nce_T=0.07):
        """
        PatchNCELoss is a custom loss function for patch-based contrastive learning.

        Args:
            batch_size (int): The batch size of the input data.
            nce_T (float, optional): The temperature parameter for the loss calculation. Defaults to 0.07.
        """
        super().__init__()
        self.batch_size = batch_size
        self.nce_T = nce_T
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def forward(self, feat_q, feat_k):
        """
        Forward pass of the PatchNCELoss.

        Args:
            feat_q (torch.Tensor): The query features.
            feat_k (torch.Tensor): The key features.

        Returns:
            torch.Tensor: The computed loss.
        """
        batchSize = feat_q.shape[0] 
        dim = feat_q.shape[1] 
        feat_k = feat_k.detach()

        l_pos = torch.bmm(feat_q.view(batchSize, 1, -1), feat_k.view(batchSize, -1, 1))
        l_pos = l_pos.view(batchSize, 1)

        batch_dim_for_bmm = self.batch_size

        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1) 
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1)) 

        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        # # continue helper
        #  for i in range(batch_size):
        #     # Selecting negative patches within the same sample
        #     neg_indices = self.get_negative_indices(num_patches, i, batch_size)
        #     negatives.append(feat_k[neg_indices])
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.nce_T

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))

        return loss    


class INSTANCENCELoss(nn.Module):
    def __init__(self, opt):
        """
        Initializes an instance of the INSTANCENCELoss class.

        Args:
            opt: An object containing options for the loss function.
        """
        super().__init__()
        self.opt = opt

        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def forward(self, feat_q, feat_k):
        """
        Computes the loss for the INSTANCENCELoss.

        Args:
            feat_q: The query features.
            feat_k: The key features.

        Returns:
            The computed loss.
        """
        batchSize = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        l_pos = torch.bmm(feat_q.view(batchSize, 1, -1), feat_k.view(batchSize, -1, 1))
        l_pos = l_pos.view(batchSize, 1)

        if self.opt.nce_includes_all_negatives_from_minibatch or self.opt.use_box:
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = self.opt.batch_size

        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.opt.nce_T

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))

        return loss

class SemNCELoss(nn.Module):
    def __init__(self, nce_T=0.07):
        """
        PatchNCELoss is a custom loss function for patch-based contrastive learning.

        Args:
            batch_size (int): The batch size of the input data.
            nce_T (float, optional): The temperature parameter for the loss calculation. Defaults to 0.07.
        """
        super().__init__()
        self.nce_T = nce_T
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def forward(self, feat_q, feat_k):
        """
        Forward pass of the PatchNCELoss.

        Args:
            feat_q (torch.Tensor): The query features.
            feat_k (torch.Tensor): The key features.

        Returns:
            torch.Tensor: The computed loss.
        """
        batchSize = feat_q.shape[0] 
        loss = []
        for mini_batch in range(batchSize):
            feat_q = feat_q[mini_batch].detach()
            dim = feat_q.shape[0] 
            feat_k = feat_k[mini_batch].detach()

            l_pos = torch.matmul(feat_q[:, 0], feat_k[:, 0])
            # l_pos = l_pos.view(batchSize, 1)

            feat_q = feat_q.view(-1, dim)
            feat_k = feat_k.view(-1, dim)
            n_neg = feat_q.size(0) 
            l_neg_cur = torch.matmul(feat_q, feat_k.transpose(2, 1)) 

            # diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
            # l_neg_curbatch.masked_fill_(diagonal, -10.0)
            l_neg = l_neg_cur.view(-1, n_neg)

            out = torch.cat((l_pos, l_neg), dim=1) / self.nce_T

            loss += self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                            device=feat_q.device))

        return (loss/batchSize).mean()
    