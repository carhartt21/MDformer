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
from helper_utils import colorEncode, find_recursive

    
def visualize_result(label):
    #TODO make universal or remove
    colors=[]
    with open('data_cfg/16Classes.json') as f:
        cls_info = json.load(f)
    for c in cls_info:
        colors.append(cls_info[c]['color'])
    # segmentation
    seg = np.asarray(label).copy()
    # seg -= 1
    seg_color = colorEncode(seg, colors)
    # aggregate images and save
    return seg_color


def remap_image(img):
    """Maps an image to a grayscale image according to the specified map and saves the result.

    Parameters
    ----------
    img : np.array (m,n,o)
        Image data with semantic segmentation.

    """
    # Read image
    colorize = False
    img_data = imageio.imread(img)
    img_name = img.split('/')[-1]
    # Check if file exists already in the output path
    # if os.path.isfile('{}/{}'.format(args.output, img_name)):
    #     return
    color_image = visualize_result(img_data)
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
    for _ in tqdm(pool.imap_unordered(remap_image, [(img) for img in imgs],
                  chunksize=args.chunk), total=len(imgs), desc='Mapping images', ascii=True):
        pass
    # Close pool
    pool.close()
