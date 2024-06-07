import os
import numpy as np
from PIL import Image

import torch
import cv2

def process_mask_list(image, mask_list, mean_threshold, stddev_threshold):
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    mask_features = []

    # calculate the color features
    for mask in mask_list:
        mean, stddev = cv2.meanStdDev(image, mask=mask)
        mask_features.append((mean, stddev))

    # initialize clustering labels
    labels = list(range(len(mask_list)))

    # merge masks with similar albedo
    for i in range(len(mask_list)):
        for j in range(i + 1, len(mask_list)):
            if labels[i] != labels[j]:
                mean1, stddev1 = mask_features[i]
                mean2, stddev2 = mask_features[j]
                if np.allclose(mean1, mean2, atol=mean_threshold) and np.allclose(stddev1, stddev2, atol=stddev_threshold):
                    target_label = labels[i]
                    for k in range(len(labels)):
                        if labels[k] == labels[j]:
                            labels[k] = target_label
    
    merged_mask_list = [np.zeros_like(mask_list[0]) for _ in range(len(mask_list))]
    for label in set(labels):
        for i, mask_label in enumerate(labels):
            if mask_label == label:
                merged_mask_list[label] = np.bitwise_or(merged_mask_list[label], mask_list[i])

    merged_mask_list = [mask for mask in merged_mask_list if not np.all(mask == 0)]
    return merged_mask_list

def refine_masks(result_path, leave_index=None):
    for k in range(10):
        if leave_index is not None:
            if k != leave_index: continue
        mask_images_dir  = f'{result_path}/ori_masks/step_00000_000{k}_rgb'
        ori_image_pth = f'{result_path}/results/step_00000_000{k}_rgb.jpg'

        items = os.listdir(mask_images_dir)
        if not items: continue

        folder_path = '/'.join(mask_images_dir.split('/')[:-2]) + f'/clean_masks/{k}'

        if not os.path.exists(folder_path):
            os.makedirs(folder_path, exist_ok=True)

        files = os.listdir(folder_path)
        # clean folder
        for file in files:
            file_path = os.path.join(folder_path, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    os.rmdir(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

        # -------   Mask Refinement in 2D Image Space   ---------
        # Get masks from segmentor
        mask_list = [np.array(Image.open(os.path.join(mask_images_dir, mask_pth))).astype(bool) for mask_pth in os.listdir(mask_images_dir) if 'full' not in mask_pth]

        masks = torch.stack([torch.from_numpy(arr) for arr in mask_list]).cuda()
        masks = masks[masks.sum((1, 2)).argsort()]
        if len(masks) == 0: continue
        image = cv2.imread(ori_image_pth)
        image = torch.tensor(image) >> 2 
        
        # Filter masks with high overlap
        for m, mask in enumerate(masks):
            union = (mask & masks[m + 1:]).sum((1, 2), True)
            masks[m + 1:] |= mask & (union > .9 * mask.sum((0, 1), True))
        
        # Identify and visualize disjoint patches
        unique, indices = masks.flatten(1).unique(return_inverse=True, dim=1)
        (cm := torch.randint(192, (unique.size(1), 3), dtype=torch.uint8))[0] = 0
        indices = indices.view_as(mask).cpu().numpy()
        unique_numbers = np.unique(indices)

        # Merge masks with similar albedo 
        mask_list = []
        for number in unique_numbers:
            if number == 0: continue
            mask = (indices == number).astype(np.uint8) * 255 
            kernel = np.ones((3,3), np.uint8)
            opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            white_pixels = np.sum(opening == 255)
            if white_pixels < 1000: continue # filter tiny region
            
            # find connected components
            n_components, output, stats, centroids = cv2.connectedComponentsWithStats(opening, connectivity=8)

            for label in range(1, n_components):  # filter tiny region
                mask = output == label
                if mask.sum() < 1000: continue
                mask_list.append((mask*255).astype('uint8'))

        mean_threshold, stddev_threshold = 10,8 # merge threshold
        mask_list_updated = process_mask_list(image, mask_list, mean_threshold, stddev_threshold)

        ori_image = Image.open(ori_image_pth).convert('RGB')
        ori_image_np = np.array(ori_image)
        # Skip white background
        mask_list_updated_new = []
        for mask in mask_list_updated:
            mask_area = mask == 255
            masked_image = ori_image_np[mask_area]
            masked_image_reshaped = masked_image.reshape(-1, 3)
            white_pixels = np.all(masked_image_reshaped == [255, 255, 255], axis=1)
            white_area_ratio = white_pixels.sum() / mask_area.sum()
            if white_area_ratio > 0.95:
                continue 
            mask_list_updated_new.append(mask)

        if len(mask_list_updated) == 0: continue 


        # -------   using SoM to mark refine masks   ---------
        if masks[0].shape[0] != ori_image_np.shape[0]:
            ori_image_np = np.array(ori_image.resize((1200,1200), Image.BICUBIC))
        
        from som.task_adapter.utils.visualizer import Visualizer
        visual = Visualizer(ori_image_np, metadata=None)

        for i, mask in enumerate(mask_list_updated_new, 1):
            img = Image.fromarray(mask)
            img.save(f'{folder_path}/{i}_mask.png')

            demo = visual.draw_binary_mask_with_number(mask, text=str(i), label_mode='1', alpha=0.05, anno_mode=['Mask', 'Mark'])

        im = demo.get_image() 
        im_pil = Image.fromarray(im)
        im_pil.save(f'{folder_path}/full.jpg')
