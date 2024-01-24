from collections import OrderedDict
import os
from data import create_dataset
from utils import save_image, translate_using_latent
from util import html
import utils
import logging
import argparse
import torch
import initialize
from visualizer import Visualizer
from data_loader import TestProvider, get_test_loader

from util.config import cfg
from data_loader import MultiDomainDataset, TrainProvider, RefProvider, get_train_loader, get_ref_loader

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
    # if os.path.exists(os.path.join(cfg.weight_path, 'config.yaml')):
        # cfg.merge_from_file(os.path.join(cfg.weight_path, 'config.yaml'))
    device = torch.device('cuda:{}'.format(cfg.TRAIN.gpu_ids[0])) if cfg.TRAIN.gpu_ids else torch.device('cpu')
    # opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    cfg.TEST.num_threads = 0   # test code only supports num_threads = 1
    cfg.TEST.batch_size = 1    # test code only supports batch_size = 1
    cfg.TEST.shuffle = False # disable data shuffling; comment this line if results on randomly chosen images are needed.
    # cfg.TESTno_flip = True    # no flip; comment this line if results on flipped images are needed.
    cfg.VISDOM.display_id = -1   # no visdom display; the test code saves the results to a HTML file.

    num_domains = len(cfg.DATASET.target_domain_names)

    # create a webpage for viewing the results
    web_dir = os.path.join(cfg.TEST.result, cfg.MODEL.name, '{}'.format(cfg.TEST.load_epoch))  # define the website directory
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s Epoch = %s' % (cfg.MODEL.name, cfg.TEST.load_epoch))
    # TODO continue here add model M
    model_G = initialize.build_model(cfg=cfg, device=device, num_domains=num_domains, mode='test')
    os.makedirs(cfg.TEST.result, exist_ok=True)
    if cfg.MODEL.weight_path is not None:
        logging.info("+ Loading Network weights from {}".format(cfg.MODEL.weight_path)) 
        for key in model_G.keys():
            file = os.path.join(cfg.MODEL.weight_path, f'{key}.pth')
            if os.path.isfile(file):
                logging.info(">> Success load weight {}".format(key))
                model_load_dict = torch.load(file, map_location=device)
                keys = model_load_dict.keys()
                values = model_load_dict.values()

                new_keys = []
                for i, mykey in enumerate(keys):
                    new_key = mykey[14:] #REMOVE 'module.'
                    new_keys.append(new_key)
                new_dict = OrderedDict(list(zip(new_keys,values)))
                model_G[key].load_state_dict(new_dict)
            else:
                logging.info(">> Does not exist {}".format(file))
    else:
        logging.info("+ No weight loaded: Couldn't find weights under {}".format(cfg.MODEL.weight_path))
# Set eval mode for every module in model_G
for _model in model_G.values():
    _model.eval().to(device)

test_loader = get_test_loader(test_dir=cfg.TEST.dir, img_size=cfg.MODEL.img_size, batch_size=cfg.TEST.batch_size, imagenet_normalize=True, shuffle=False)
test_provider = TestProvider(test_loader, 'test')
for i, data in enumerate(test_loader):
    if i >= cfg.TEST.num_images:  # only apply our model to opt.num_test images.
        break
    sample = next(test_provider)
    logging.info("Processing image {}/{}".format(i, test_loader.__len__()))
    for d_trg in cfg.TEST.target_domains:
        d_trg = utils.domain_to_onehot(d_trg, cfg.TEST.target_domains)
        translate_using_latent(model_G=model_G, cfg=cfg, input=sample, d_trg=d_trg, psi=cfg.TEST.psi, filename='output.png', latent_dim=cfg.MODEL.latent_dim, feat_layers=cfg.MODEL.feat_layers)