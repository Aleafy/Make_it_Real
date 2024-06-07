# --------------------------------------------------------
# Semantic-SAM: Segment and Recognize Anything at Any Granularity
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Hao Zhang (hzhangcx@connect.ust.hk)
# --------------------------------------------------------

import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from .automatic_mask_generator import SemanticSamAutomaticMaskGenerator
import os

def get_white_ratio(image_ori, mask):
    image_mask_sum = image_ori[mask].sum(-1)
    count_white = np.sum(image_mask_sum==765)
    percentage = (count_white / len(image_mask_sum))
    if percentage > 0.80:
        return True
    else:
        return False

def inference_semsam_auto(model, image, level, all_classes, all_parts, thresh, text_size, hole_scale, island_scale, semantic, refimg=None, reftxt=None, audio_pth=None, video_pth=None, label_mode='1', alpha=0.1, anno_mode=['Mask'], points_per_side=32, save_dir=None):
    t = []
    t.append(transforms.Resize(int(text_size), interpolation=Image.BICUBIC))
    transform1 = transforms.Compose(t)
    image_ori = transform1(image)

    image_ori = np.asarray(image_ori)
    images = torch.from_numpy(image_ori.copy()).permute(2,0,1).cuda()

    mask_generator = SemanticSamAutomaticMaskGenerator(model,points_per_side=points_per_side,
            pred_iou_thresh=0.88, 
            stability_score_thresh=0.92,
            min_mask_region_area=10,
            level=level,
        )
    outputs = mask_generator.generate(images)

    sorted_anns = sorted(outputs, key=(lambda x: x['area']), reverse=True)

    folder_path = save_dir
    files = os.listdir(folder_path)
    for file in files:
        file_path = os.path.join(folder_path, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                os.rmdir(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")
  
    for i, ann in enumerate(sorted_anns):
        mask = ann['segmentation']
        if get_white_ratio(image_ori, mask):
            continue
        gray_mask = np.uint8(mask) * 255
        image = Image.fromarray(gray_mask)
        image.save(f"{save_dir}/{i}.png")

    return sorted_anns
