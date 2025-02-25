from os import walk, path
import fnmatch
import numpy as np

def find_recursive(root_dir, ext='.jpg', names_only=False):
    files = []
    for root, dirnames, filenames in walk(root_dir):
        for filename in fnmatch.filter(filenames, '*' + ext):
            if names_only:
                files.append(filename)
            else:
                files.append(path.join(root, filename))
    return files

def colorEncode(labelmap, colors, mode='RGB'):
    labelmap = labelmap.astype('uint8')
    labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3),
                            dtype=np.uint8)
    labels = np.unique(labelmap)
    if len(labels) > len(colors) + 1:
        print('Error: labelmap contains too many labels - {}'.format(labelmap.max()))
    for label in labels:
        if label < 0:
            continue
        if label == 255:
            labelmap_rgb += (labelmap == label)[:, :, np.newaxis] * \
            np.tile((0, 0, 0),
                    (labelmap.shape[0], labelmap.shape[1], 1)).astype(np.uint8)        
        else:
            labelmap_rgb += (labelmap == label)[:, :, np.newaxis] * \
            np.tile(colors[label],
                    (labelmap.shape[0], labelmap.shape[1], 1)).astype(np.uint8)

    if mode == 'BGR':
        return labelmap_rgb[:, :, ::-1]
    else:
        return labelmap_rgb