
import numpy as np
import cv2
from PIL import Image
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import json
import argparse
from helper_utils import colorEncode, find_recursive
from os.path import isdir

def map_segmentation_to_grid(segmentation_map, patch_size, min_coverage):
    """
    Maps a segmentation map onto a grid of cells.

    Args:
        segmentation_map (numpy.ndarray): A 2D numpy array representing the segmentation map.
        m (int): The number of cells in each row/column of the grid.
        min_coverage (float): The minimum percentage of pixels that should be covered by a class in a cell for the class to be assigned to the cell.

    Returns:
        numpy.ndarray: A 2D numpy array representing the grid with the segmentation map mapped onto it.
    """
    height, width = segmentation_map.shape

    assert height % patch_size == 0, "Segmenation map height {} is not divisible by patch size {}".format(height, patch_size)
    assert width % patch_size == 0, "Segmenation map width {} is not divisible by patch size {}".format(width, patch_size)

    n_patches = (height // patch_size) * (width // patch_size)
    
    # Get list of unique classes in the segmentation map
    _classes = np.unique(segmentation_map)

    # Calculate the minimum number of pixels that should be covered by the segmentation map in each patch
    min_pixels = patch_size ** 2 * min_coverage
    
    # Create an empty grid to map the segmentation onto
    grid = np.full((height, width), 255)
    # empty_array = np.full((height, width), 255)
    
    # Iterate over each cell in the grid
    for i in range(n_patches):
        for j in range(n_patches):
            # Calculate the bounds of the current cell
            top = i * patch_size
            bottom = (i + 1) * patch_size
            left = j * patch_size
            right = (j + 1) * patch_size
            
            # Iterate over each class in the segmentation map
            for c in _classes:
                # Count the number of pixels covered by the current class in the current cell
                pixels_covered = np.sum(segmentation_map[top:bottom, left:right] == c)
                
                # If the number of pixels covered is greater than or equal to the minimum required, assign the class to the current cell
                if pixels_covered >= min_pixels:
                    grid[top:bottom, left:right] = c
    
    return grid


def create_grid_view(segmentation_map, patch_size=16, min_coverage=0.9):
    # Create a new image with a white background
    height, width = segmentation_map.height, segmentation_map.width
    img = Image.new('RGB', (width, height), color='white')

    # Draw the segmentation map on the image
    # segmentation_map_pil = Image.fromarray(color_encode(segmentation_map))
    img.paste(segmentation_map, (0, 0))

    # Draw a grid on the image
    draw = ImageDraw.Draw(img)
    for x in range(0, width, patch_size):
        draw.line((x, 0, x, height), fill='gray')
    for y in range(0, height, patch_size):
        draw.line((0, y, width, y), fill='gray')

    # Display the image
    return img



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
    # Read args
    args = parser.parse_args()
    # Generate image list
    if isdir(args.input):
        imgs = find_recursive(args.input, ext='.png')
    # for path in imgs:
    path = "/media/chge7185/HDD1/datasets/MDformer/seg_labels/ADEout/ADE_train_00000004.png"
    print ('path: {}'.format(path))
    # Load the image from the path
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    print ('segmentation_map: {}'.format(img))

    # Convert the image to a numpy array
    segmentation_map = np.array(img)

    # Define the colors for each class
    colors=[]
    with open('data_cfg/16Classes.json') as f:
        cls_info = json.load(f)
    for c in cls_info:
        colors.append(cls_info[c]['color'])

    grid_map = map_segmentation_to_grid(segmentation_map, 8, 0.9)
    print ('grid_map: {} {}'.format(grid_map.size, grid_map))

    # original_image = Image.fromarray(img)

    # Display the segmentation map and grid map side by side
    segmentation_map_pil = Image.fromarray(colorEncode(segmentation_map, colors))

    grid_map_pil = Image.fromarray(colorEncode(grid_map.astype(np.uint8), colors))
    # create_grid_view(segmentation_map)
    side_by_side = Image.new('RGB', (segmentation_map_pil.width + grid_map_pil.width, segmentation_map_pil.height))
    # side_by_side.paste(original_image, (0, 0))
    side_by_side.paste(create_grid_view(segmentation_map_pil), (0, 0))
    side_by_side.paste(create_grid_view(grid_map_pil), (segmentation_map_pil.width, 0))
    side_by_side.show()
    input("Press Enter to continue...")
