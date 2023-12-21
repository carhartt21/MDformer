# import imageio
import numpy as np
import argparse
import os
import json
import multiprocessing as mp
from scipy.io import loadmat
from timeit import default_timer as timer
from tqdm import tqdm
from pathlib import Path
# internal libraries
# from ..utils import colorEncode, find_recursive
import argparse
import json
import fnmatch
import imageio.v2 as imageio

def find_recursive(root_dir, ext='.jpg', names_only=False):
    files = []
    for root, dirnames, filenames in os.walk(root_dir):
        for filename in fnmatch.filter(filenames, '*' + ext):
            if names_only:
                files.append(filename)
            else:
                files.append(os.path.join(root, filename))
    return files

def visualize_result(label):
    #TODO make universal or remove
    colors=[]
    with open('data_cfg/outside15k.json') as f:
        cls_info = json.load(f)
    for c in cls_info:
        colors.append(cls_info[c]['color'])
    # segmentation
    seg = np.asarray(label).copy()
    # seg -= 1
    seg_color = colorEncode(seg, colors)
    # aggregate images and save
    return seg_color

def remap_image_mat(img):
    """Maps an image to a grayscale image according to the specified map and saves the result.

    Parameters
    ----------
    img : np.array (m,n,o)
        Image data with semantic segmentation.

    """
    # Read image
    colorize = False
    img_data = imageio.imread(img)
    if args.dataset == 'mapillary':
        # print('img_data.shape: {}'.format(img_data.shape))
        img_data = np.delete(img_data, 3, 1)
        unique_values = np.unique(img_data.reshape(-1, 3), axis=0)
    elif args.dataset == 'ADE20k':
        unique_values = np.unique(img_data.reshape(-1, 1), axis=0)
    elif args.dataset == 'Cityscapes':
        unique_values = np.unique(img_data.reshape(-1, 1), axis=0)
    else:
        print('Exception: Dataset type {} unknown'.format(args.dataset))
        # ignore label if it doesn't contain sky
        # if 3 not in unique_values: 
        #     return
    gray_image = np.zeros((img_data.shape[0], img_data.shape[1]), dtype='uint8')
    img_name = img.split('/')[-1]
    # Check if file exists already in the output path
    if os.path.isfile('{}/{}'.format(args.output, img_name)):
        return
    for val in unique_values:
        if args.dataset == 'mapillary':
            try:
                old_class = mapColors.index(list(val))
                gray_image += ((img_data == val).all(axis=2) * mapNames[str(old_class)]).astype(np.uint8)
            except ValueError:
                print('Exception: class {} not found'.format(val))
        elif args.dataset == 'ADE20k':
            gray_image += ((img_data == val) * mapNames[str(int(val))]).astype(np.uint8)
            pass
        elif args.dataset == 'Cityscapes':
            gray_image += ((img_data == val) * mapNames[str(int(val))]).astype(np.uint8)  
        else:
            print('Exception: Dataset type {} unknown'.format(args.dataset))                     
    imageio.imwrite('{}/{}'.format(args.output, img.split('/')[-1]), gray_image)
    if colorize:
        color_image = visualize_result(gray_image)
        imageio.imwrite('{}/{}'.format(args.output, img.split('/')[-1].replace('.png', '_colored.png')), color_image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Maps and converts a segmentation image dataset to grayscale images"
    )
    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="Image path, or a directory name"
    )
    parser.add_argument(
        "--output",
        required=False,
        type=str,
        help="Path for output folder",
        default='output/'
    )
    parser.add_argument(
        "--dataset",
        required=False,
        type=str,
        help="Dataset type",
        default='mapillary'
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
        default=1
    )
    # Read args
    args = parser.parse_args()
    # Load the label mapping from a JSON file
    if args.dataset == 'mapillary':
        colorMappingFile = Path('data_cfg/colorsMapillary.json')
        mappingFile = Path('data_cfg/MapillaryMap16.json')
        with open(colorMappingFile) as mfile:
            mapColors = list(json.load(mfile).values())
        with open(mappingFile) as mfile:
            mapNames = json.load(mfile)
    elif args.dataset == 'City':
        mappingFile = Path('data_cfg/CityscapesMap16.json')
        with open(mappingFile) as mfile:
            mapNames = json.load(mfile)
    elif args.dataset == 'PASCAL':
        mappingFile = Path('data_cfg/PASCALMap.json')
        with open(mappingFile) as mfile:
            mapNames = json.load(mfile)
    elif args.dataset == 'ADE20k':
        mappingFile = Path('data_cfg/ADEMap16.json')
        with open(mappingFile) as mfile:
            mapNames = json.load(mfile)
    else:
        print('Exception: Dataset type {} unknown'.format(args.dataset))

    # Generate image list
    if os.path.isdir(args.input):
        print(args.input)
        imgs = find_recursive(args.input, ext='.png')
    else:
        imgs = [args.input]
    assert len(imgs), "Exception: imgs should be a path to image (.jpg) or directory."
    # Create output directory
    if not os.path.isdir(args.output):
        print('Creating empty output directory: {}'.format(args.output))
        os.makedirs(args.output)
    # Create worker pool
    pool = mp.Pool(args.nproc)
    # Assign tasks to workers
    for _ in tqdm(pool.imap_unordered(remap_image_mat, [(img) for img in imgs],
                  chunksize=args.chunk), total=len(imgs), desc='Mapping images', ascii=True):
        pass
    # Close pool
    pool.close()
