from os import listdir, walk, path, makedirs, mkdir
import os
from pathlib import Path
import re
from fnmatch import filter
import glob
import shutil
import argparse
from tqdm import tqdm
import multiprocessing as mp
# import pandas as pd
import numpy as np
import clip
from PIL import Image
import torch

# file = '/home/chge7185/repositories/weather_dataset/skyfinder_complete_table.csv'
# file = '/home/chge7185/repositories/weather_dataset/all_attributes.csv'
# source_path = '/media/chge7185/HDD1/datasets/skyfinder/webcam-labeler'

# tae_attributes = np.loadtxt('/media/chge7185/HDD1/datasets/transient_attributes/annotations/attributes.txt', dtype=str)
# target_attributes = ['daylight', 'night', 'sunrisesunset', 'dawndusk', 'spring', 'summer', 'autumn', 'winter', 'sunny', 'fog', 'clouds', ['storm', 'snow', 'ice', 'cold'], ['sunny', 'dry', 'warm']]
# target_attributes = ['daylight', 'sunrisesunset', 'dawndusk', 'night', 'spring', 'summer', 'autumn', 'winter', 'sunny', 'snow', 'rain', 'fog']
target_attributes = ['summer', 'autumn', 'winter', 'spring']

threshold = 0.5


def find_recursive(root_dir, ext='.jpg', names_only=False):
    files = []
    for root, dirnames, filenames in walk(root_dir):
        for filename in filter(filenames, '*' + ext):
            if names_only:
                files.append(filename)
            else:
                files.append(path.join(root, filename))
    return files


def save_image(x, condition, file):
    if x[condition]:
        file_path = glob.glob(source_path + "/images/**/" + x[file], recursive = True)
        shutil.copy(file_path[0], source_path + "/images_by_attributes/" + condition)
    return

def sort_images_by_label(input_path):
    in_array = np.load(input_path)
    attributes = (in_array > threshold).nonzero()[0]
    for idx in attributes:
        input_path = Path(input_path)
        # _path = input_path.replace('attributes', 'images')
        _path, _name = str(input_path.parent), input_path.name
        image_path = _path.replace('domain_labels', 'images')

        image_name = _name.replace('npy', 'png')
        # print(path.join(image_path, image_name))
        if not path.isfile(path.join(image_path, image_name)):
            image_name = image_name.replace('png', 'jpg')
            # print(path.join(image_path, image_name))
            if not path.isfile(path.join(image_path, image_name)):
                # print('File {} not found'.format(str(image_path) + '/'+ image_name))
                return
        # parent, name = str(image_path.parent), image_path.name
        # domain_path = os.path.join(parent.replace('domain', 'domain_labels'), 
        # name.replace('.jpg', '.npy')).replace('.png', '.npy')            
        
        out_dir_imgs = '{}/images/{}'.format(args.output, target_attributes[idx])
        out_dir_atts = '{}/domain_labels/{}'.format(args.output, target_attributes[idx])
        out_img_path = '{}/{}'.format(out_dir_imgs, image_name)
        out_att_path = '{}/{}'.format(out_dir_atts, _name)

        # out_dir = '{}/{}{}/'.format(args.output, target_attributes[idx], str(image_path).split('images')[-1])
        # out_img_path = '{}{}'.format(out_dir, image_name)
        
        if not path.isdir(out_dir_imgs):
            try:
                makedirs(out_dir_imgs, exist_ok=True)
            except FileExistsError as e:
                print(str(e), out_dir_imgs)
        shutil.copy('{}/{}'.format(image_path, image_name), out_img_path)
        if not path.isdir(out_dir_atts):
            try:
                makedirs(out_dir_atts, exist_ok=True)
            except FileExistsError as e:
                print(str(e), out_dir_atts)                
        shutil.copy(input_path, out_att_path)
    return

def sort_images_using_clip(input_path, model, text, preprocess, device):
    image = preprocess(Image.open(input_path)).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()     
    # print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]] 
    # print((probs > threshold).nonzero()[1])        
    attributes = (probs > threshold).nonzero()[1]
    # attributes = torch.where(probs.squeeze() > threshold).cpu().numpy()
    for idx in attributes:
        input_path = Path(input_path)
        # _path = input_path.replace('attributes', 'images')
        _, _name = str(input_path.parent), input_path.name           
        
        out_dir_imgs = '{}/images/{}'.format(args.output, target_attributes[idx])
        # out_dir_atts = '{}/domain_labels/{}'.format(args.output, target_attributes[idx])
        out_img_path = '{}/{}'.format(out_dir_imgs, _name)
        # out_att_path = '{}/{}'.format(out_dir_atts, _name)

        # out_dir = '{}/{}{}/'.format(args.output, target_attributes[idx], str(image_path).split('images')[-1])
        # out_img_path = '{}{}'.format(out_dir, image_name)
        
        if not path.isdir(out_dir_imgs):
            try:
                makedirs(out_dir_imgs, exist_ok=True)
            except FileExistsError as e:
                print(str(e), out_dir_imgs)
        shutil.copy(input_path, out_img_path)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Sorts images by attributes from a numpy array"
    )
    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="Image path, or a directory name"
    )
    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="Path for output files",
    )
    parser.add_argument(
        "--nproc",
        required=False,
        type=int,
        help="Number of parralel processes",
        default=mp.cpu_count()
    )
    parser.add_argument(
        "--chunk",
        required=False,
        type=int,
        help="Chunk size for each worker thread",
        default=mp.cpu_count()
    )
    # Read args
    args = parser.parse_args()
    if not path.isdir(args.output):
        print('Creating empty output directory {}'.format(args.output))
        makedirs(args.output)
        for attribute in target_attributes:
            mkdir('{}/{}'.format(args.output, attribute))
    else:
        for attribute in target_attributes:
            if not path.isdir('{}/{}'.format(args.output, attribute)):
                mkdir('{}/{}'.format(args.output, attribute))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # classes = ["autumn", "summer", "winter", "spring"]
    text = clip.tokenize(target_attributes).to(device)
    # text = torch.cat([clip.tokenize(f"a {c} scene") for c in classes]).to(device)

    # text = clip.tokenize(["autumn", "summer", "winter", "spring"]).to(device)
          
    
    # Generate file list
    if path.isdir(args.input):
        print(args.input)
        files = find_recursive(args.input, ext='.png')
        assert len(files), "Exception: files should be a path to image csv file or directory."
        print('Found {} files'.format(len(files)))
        # sort_images_by_label(files[223])
        for _file in tqdm(files, desc='Sorting images by label', ascii=True):
            sort_images_using_clip(_file, model, text, preprocess, device)
        # pool = mp.Pool(args.nproc)
        # for _ in tqdm(pool.imap_unordered(sort_images_by_label, [(_file) for _file in files], chunksize=args.chunk),
        #               total=len(files), desc='Sorting images by label', ascii=True):
        #     pass
        # ## Close pool
        # pool.close()
    else:
        print('Input is not a directory!')

       