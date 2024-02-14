from data_loader import MultiDomainDataset, TrainProvider, get_train_loader, get_ref_loader, RefProvider
from PIL import Image
from PIL import Image, ImageDraw
import torch
from einops import rearrange, repeat, asnumpy
from utils import tensor2im
# matplotlib.use('Agg')

import numpy as np
from PIL import Image, ImageDraw
import numpy as np
import sys
import os
import time
import json
from helper.helper_utils import colorEncode
from util.custom_transforms import UnNormalize
import torchvision.transforms as transforms
from torchvision.utils import save_image

#TODO: add semantic embedding

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

# IMG_MEAN = [0.5, 0.5, 0.5]
# IMG_STD = [0.5, 0.5, 0.5]

attributes = ['daylight', 'sunrisesunset', 'dawndusk', 'night', 'spring', 'summer', 'autumn', 'winter', 'sunny', 'snow', 'rain', 'fog']
    
def delete_files_without_classes(folder_path, classes):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            seg_map = np.asarray(Image.open(file_path))
            print("seg_map shape: ", seg_map)
            time.sleep(1)  # Wait for 5 seconds
            contains_classes = any(class_name in seg_map for class_name in classes)
            if not contains_classes:
                print("Deleting file: ", file_path)
                # os.remove(file_path)

def denormalize(x, mean=IMG_MEAN, std=IMG_STD):
    # 3, H, W, B
    ten = x.clone().permute(1, 2, 3, 0)
    for t, m, s in zip(ten, mean, std):
        t.mul_(s).add_(m)
    # B, 3, H, W
    return torch.clamp(ten, 0, 1).permute(3, 0, 1, 2)

# def denormalize(x):
#     out = (x + 1) / 2
#     return out.clamp_(0, 1)

def colorize_result(seg_gray):
    colors=[]
    with open('helper/data_cfg/16Classes.json') as f:
        cls_info = json.load(f)
    for c in cls_info:
        colors.append(cls_info[c]['color'])
    return colorEncode(np.asarray(seg_gray), colors)

def visualize_image(input, ref, patch_size=8):
    # Convert torch.Tensor to numpy array
    img_src = input.img_src.cpu()
    img_1 = denormalize(img_src)

    # img_3 = UnNormalize(IMG_MEAN, IMG_STD)(img_src)

    # print("denormalize: {} denorm: {} unnormalize: {}".format(img_1[0,0,:10], img_2[0,0,:10], img_3[0,0,:10]))


    save_image(img_src, 'output/img_src.png')
    save_image(img_1, 'output/denormalize.png')
    # save_image(UnNormalize(IMG_MEAN, IMG_STD)(img_src), 'unnormalize.png')

    # img_1 = Image.fromarray(tensor2im(img_1), 'RGB')
    # img_2 = transforms.functional.to_pil_image(img_2[0])
    # print("img_1 RGB: ", np.asarray(img_1)[0, 0, :10])
    # print("img_5 RGB: ", np.asarray(img_2)[0, 0, :10])
    pil_img = transforms.ToPILImage()(img_1[0])
    pil_img.save('output/img_src_2.png')

    seg_mask = rearrange(input.seg, 'b (h w) -> b h w', h=int(input.seg.shape[1]**(1/2)))
    seg_mask = repeat(seg_mask, 'b h w -> b (h h1) (w w1)', h1 = patch_size, w1 = patch_size)
    seg_mask = asnumpy(seg_mask[0].cpu())
    box = input.bbox.cpu().numpy()
    domain = input.d_src.cpu().numpy()
    visualization = Image.new('RGB', (pil_img.size[0] * 3, pil_img.size[1]))

    # Paste the original image on the left side
    visualization.paste(pil_img, (0, 0))

    # Paste the reference image on the right side
    ref_img = denormalize(ref.img_ref.cpu())
    # ref_img = tensor2im(ref_img)
    save_image(ref_img, 'output/ref_img_.png')
    pil_img_ref = transforms.ToPILImage()(ref_img[0])
    visualization.paste(pil_img_ref, (pil_img.size[1]*2, 0))

    # Paste the segmented mask on the right side
    seg_mask = colorize_result(seg_mask)
    seg_mask_pil = Image.fromarray(seg_mask, 'RGB')

    seg_mask_pil.save('output/seg_mask_color.png')
    visualization.paste(seg_mask_pil, (pil_img.size[1], 0))
    # Draw bounding boxes on the visualization
    draw = ImageDraw.Draw(visualization)
    # boxes = _box
    if len(box[0]) > 0:
        for bbox in box[0]:
            x1, y1, x2, y2 = bbox
            draw.rectangle([(x1, y1), (x2, y2)], outline=(255, 255, 0), width=2)
    # if domain.sum() > 0:
    #     idx = np.argmax(domain[:, 4:8], axis=1)
    #     draw.text((0, 0), attributes[idx.item() + 4], fill=(255, 0, 0))
    # trg_domain = input.d_trg.cpu().numpy() 
    # ref_idx = np.argmax(trg_domain[4:8], axis=1)
    visualization.save('output/test.png')
    # draw.text((img.shape[1]*2, 0), attributes[ref_idx.item() + 4], fill=(255, 0, 0))


def test_multidomaindataset(batch_size=1, img_size=(352, 352)): 
    # delete_files_without_classes(folder_path="/media/chge7185/HDD1/datasets/MDformer/seg_labels/ADEout", classes=[8, 10, 11, 13])
    train_loader = get_train_loader(img_size=img_size,
        imagenet_normalize=True,
        batch_size=batch_size,
        train_dir='train_dir.txt',
        num_workers=0, 
        num_domains=12)
    ref_loader = get_ref_loader(img_size=img_size,
        imagenet_normalize=True,
        batch_size=batch_size,
        ref_list='ref_list.txt', 
        num_workers=0,
        num_domains=12)
    latent_dim = 16
    fetcher = TrainProvider(train_loader, mode='train')
    ref_fetcher = RefProvider(ref_loader, mode='train')
    # while True:
    while True:
        inputs = next(fetcher)
        ref = next(ref_fetcher)
        visualize_image(inputs, ref)
        print('Press a key to continue...')
        sys.stdin.read(1)  # Wait for a key press

if __name__ == '__main__':
    test_multidomaindataset()
