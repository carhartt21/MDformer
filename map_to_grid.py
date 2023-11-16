
import numpy as np
import cv2
from PIL import Image
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

def map_segmentation_to_grid(segmentation_map, m, min_coverage):
    """
    Maps a segmentation map onto a grid of cells.

    Args:
        segmentation_map (numpy.ndarray): A 2D numpy array representing the segmentation map.
        m (int): The number of cells in each row/column of the grid.
        min_coverage (float): The minimum percentage of pixels that should be covered by a class in a cell for the class to be assigned to the cell.

    Returns:
        numpy.ndarray: A 2D numpy array representing the grid with the segmentation map mapped onto it.
    """
    
    # Get the dimensions of the segmentation map
    height, width = segmentation_map.shape
    
    # Calculate the size of each grid cell
    cell_size = height // m
    
    # Calculate the number of cells in the grid
    n_cells = m * m

    _classes = np.unique(segmentation_map)
    print('_classes: {}'.format(_classes))
    
    # Calculate the total number of pixels in the segmentation map
    cell_pixels = cell_size * cell_size
    
    # Calculate the minimum number of pixels that should be covered by the segmentation map in each cell of the grid
    min_pixels = cell_pixels * min_coverage
    
    # Create an empty grid to map the segmentation onto
    grid = np.full((height, width), 255)
    # empty_array = np.full((height, width), 255)
    
    # Iterate over each cell in the grid
    for i in range(m):
        for j in range(m):
            # Calculate the bounds of the current cell
            top = i * cell_size
            bottom = (i + 1) * cell_size
            left = j * cell_size
            right = (j + 1) * cell_size
            
            # Iterate over each class in the segmentation map
            for c in _classes:
                # Count the number of pixels covered by the current class in the current cell
                pixels_covered = np.sum(segmentation_map[top:bottom, left:right] == c)
                
                # If the number of pixels covered is greater than or equal to the minimum required, assign the class to the current cell
                if pixels_covered >= min_pixels:
                    grid[top:bottom, left:right] = c
    
    return grid

def color_encode(segmentation_map):
    # Define the colors for each class
    colors = {
        0: [0, 0, 0],        # background
        1: [128, 0, 0],      # aeroplane
        2: [0, 128, 0],      # bicycle
        3: [128, 128, 0],    # bird
        4: [0, 0, 128],      # boat
        5: [128, 0, 128],    # bottle
        6: [0, 128, 128],    # bus
        7: [128, 128, 128],  # car
        8: [64, 0, 0],       # cat
        9: [192, 0, 0],      # chair
        10: [64, 128, 0],    # cow
        11: [192, 128, 0],   # diningtable
        12: [64, 0, 128],    # dog
        13: [192, 0, 128],   # horse
        14: [64, 128, 128],  # motorbike
        15: [192, 128, 128], # person
        16: [0, 64, 0],      # potted plant
        17: [128, 64, 0],    # sheep
        18: [0, 192, 0],     # sofa
        19: [128, 192, 0],   # train
        20: [0, 64, 128],    # tv/monitor
        21: [224, 224, 192], # wall
        22: [224, 224, 0],   # window
        23: [224, 0, 224],   # wood
        24: [0, 224, 224],   # person (difficult)
        25: [224, 0, 0]      # car (difficult)
    }
    
    # Create an empty color-encoded image
    height, width = segmentation_map.shape
    color_encoded = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Iterate over each class in the segmentation map and assign the corresponding color to each pixel
    for c in colors:
        color_encoded[segmentation_map == c] = colors[c]
    
    return color_encoded

def create_grid_view(segmentation_map, m=16, min_coverage=0.9):
    # Create a new image with a white background
    height, width = segmentation_map.height, segmentation_map.width
    img = Image.new('RGB', (width, height), color='white')
    cell_size = height // m

    # Draw the segmentation map on the image
    # segmentation_map_pil = Image.fromarray(color_encode(segmentation_map))
    img.paste(segmentation_map, (0, 0))

    # Draw a grid on the image
    draw = ImageDraw.Draw(img)
    for x in range(0, width, cell_size):
        draw.line((x, 0, x, height), fill='gray')
    for y in range(0, height, cell_size):
        draw.line((0, y, width, y), fill='gray')

    # Display the image
    return img



if __name__ == '__main__':

    path = 'segmentation_map.png'
    # Load the image from the path
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    print ('segmentation_map: {}'.format(img))
   
    # Convert the image to a numpy array
    segmentation_map = np.array(img)

    grid_map = map_segmentation_to_grid(segmentation_map, 16, 0.9)
    print ('grid_map: {} {}'.format(grid_map.size, grid_map))

    # Display the segmentation map and grid map side by side
    segmentation_map_pil = Image.fromarray(color_encode(segmentation_map))

    grid_map_pil = Image.fromarray(color_encode(grid_map.astype(np.uint8)))
    # create_grid_view(segmentation_map)
    side_by_side = Image.new('RGB', (segmentation_map_pil.width + grid_map_pil.width, segmentation_map_pil.height))
    side_by_side.paste(create_grid_view(segmentation_map_pil), (0, 0))
    side_by_side.paste(create_grid_view(grid_map_pil), (segmentation_map_pil.width, 0))
    side_by_side.show()
