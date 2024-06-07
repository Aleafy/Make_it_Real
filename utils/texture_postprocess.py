import re
import os
from PIL import Image

import numpy as np

import cv2


from tqdm import tqdm
from scipy.spatial import cKDTree


# ------  I. Region-level texture map partitioning  ------  
def region_refine(masks_dir = 'experiments/obj_train/masks', albedo_pth = 'textures/Image_0.png', pixel=1024):
    """
    Refine masks in texture space based on block-centric clustering by albedo.
    
    Args:
        masks_dir (str): Directory of initial material regions acquired by back-projection and uv unwrapping.
        albedo_pth (str): Path of albedo texture map.
        pixel (int): Pixel size of albedo texture map.
    Returns:
        masks_refine_dir: Directory of refined material regions.
    """
    divide_masks(masks_dir)
    mat_mask_pths = [os.path.join(masks_dir, f) for f in os.listdir(masks_dir)] 

    albedo_map = np.array(Image.open(albedo_pth))
    
    size = (pixel, pixel)
    union_mask = np.zeros(size, dtype=bool) 

    mask_list = [] # regions with the assigned material
    albedo_avgs = {} # average albedo(representative value) of each region   
    
    for i, mask_pth in enumerate(mat_mask_pths):
        mask_i = Image.open(mask_pth)
        mask = (np.array(mask_i) == 255)
        if mask.shape[0] != union_mask.shape[0]:
             mask_i = mask_i.resize(size, Image.BICUBIC)
             mask = (np.array(mask_i) == 255)
        mask_list.append(mask)
        union_mask |= mask
        selected_albedo = albedo_map[mask]
        if selected_albedo.shape[0] == 0: continue
        average_value = np.mean(selected_albedo)

        albedo_avgs[i] = average_value

    print(f"Average albedo of each region: {albedo_avgs}")

    # get missing region
    un_mask = ~union_mask 
    
    # complete missing part of material regions
    for i in range(size[0]):
        for j in range(size[1]):
            if un_mask[i][j] == True:
                color_mean = np.mean(albedo_map[i,j])
                index = find_closest_key(albedo_avgs, color_mean)
                mask_list[index][i][j] = True
        
    # save refined material regions
    mat_refine_list = [mat_pth.replace('/masks', '/masks_refine') for mat_pth in mat_mask_pths]
    if mat_refine_list != []:
        masks_refine_dir = os.path.split(mat_refine_list[0])[0]
        os.makedirs(os.path.split(mat_refine_list[0])[0], exist_ok=True)
    
    indexed_areas = [(index, np.sum(mask)) for index, mask in enumerate(mask_list)]
    sorted_indexed_areas = sorted(indexed_areas, key=lambda x: x[1], reverse=True)
    mask_list = [mask_list[index] for index, area in sorted_indexed_areas]
    mat_refine_list = [mat_refine_list[index] for index, area in sorted_indexed_areas]

    for i in range(len(mask_list)):
        mask = mask_list[i] * 255
        mask_im = Image.fromarray(mask.astype('uint8'),'L')
        mask_pth = mat_refine_list[i]
        mask_im.save(mask_pth)

    return masks_refine_dir


# ------  II. Pixel-level albedo-referenced SVBRDF estimation  ------  
def pixel_estimate(masks_refine_dir = 'experiments/obj_train/masks_refine', albedo_pth = 'experiments/obj/mesh/albedo.png', pixel=1024):
    """
    Index key albedo referenced by query albedo values, and store the indexed different material maps separately in dir 'mat_parts_pbr'.
    
    Args:
        masks_refine_dir (str): Directory of region-level completed and refined texture partitions.
        albedo_pth (str): Path of albedo texture map.
        pixel (int): Pixel size of albedo texture map.
    Returns:
        generate_pbr_dir: Directory of final material texture maps.
    """
    mat_masks = [os.path.join(masks_refine_dir, name) for name in os.listdir(masks_refine_dir) if 'postde.png' in name]
    save_dir = masks_refine_dir.replace('masks_refine', 'mat_parts_pbr')
    os.makedirs(save_dir, exist_ok=True)

    for mat_mask_pth in tqdm(mat_masks): # for each material mask
        mask = Image.open(mat_mask_pth).resize((pixel, pixel), Image.BICUBIC)
        mask = (np.array(mask) == 255)
        
        # search by name from material library
        mat_name = os.path.split(mat_mask_pth)[-1]
        main_type = mat_name.split('_')[0]
        sub_type = mat_name.split('_')[1]
        sub_type_new = format_string(sub_type)
        material_dir = os.path.join('data/material_lib/pbr_maps/train', main_type, sub_type_new) 

        # query/ori albedo
        A = q_albedo = np.array(convert_cv2_to_pil(mask_hist(albedo_pth, mat_mask_pth, pixel)))

        # key albedo
        k_albedo_pth = f'{material_dir}/basecolor.png'
        B = k_albedo = np.array(convert_cv2_to_pil(normal_hist(k_albedo_pth, pixel)))
        
        # KD-Tree algorithm to accelerate searching
        B_reshaped = B.reshape(-1, 3)
        kd_tree = cKDTree(B_reshaped) 

        A_masked = A[mask == 1] 
        A_masked_reshaped = A_masked.reshape(-1, 3)

        # Find the closest color index in key albedo(B) referenced by query albedo(A)
        _, indices_masked = kd_tree.query(A_masked_reshaped, workers=-1)

        valid_indices = indices_masked[indices_masked < B_reshaped.shape[0]]

        if len(valid_indices) < len(indices_masked):
            print("Warning: Some indices are out of bounds and will be discarded.")

        indices_2d_masked = np.full(mask.shape, -1, dtype=object)  
        valid_rows, valid_cols = np.where(mask == 1)
        valid_rows = valid_rows[indices_masked < pixel * pixel]
        valid_cols = valid_cols[indices_masked < pixel * pixel]
        indices_rows, indices_cols = np.unravel_index(valid_indices, (pixel, pixel))

        for i in range(len(valid_indices)):
            indices_2d_masked[valid_rows[i], valid_cols[i]] = (indices_rows[i], indices_cols[i])

        # Estimate svbrdf values for each material mask
        mat_pngs = [os.path.join(material_dir, name) for name in os.listdir(material_dir) if '.png' in name]

        for im_pth in mat_pngs:
            t_image = Image.open(im_pth)
            t_image = t_image.resize((pixel, pixel), Image.BICUBIC)

            if ('height' in im_pth) or ('displacement') in im_pth:
                t_image = np.array(t_image.convert('I'))
            else:
                t_image = np.array(t_image.convert('RGB'))
            t_image_index = np.zeros_like(t_image)
            for i in range(pixel):
                for j in range(pixel):
                    if not mask[i][j]: continue 
                    t_image_index[i][j] = t_image[indices_2d_masked[i][j]]

            im_type = os.path.split(im_pth)[-1]
            Image.fromarray(t_image_index).save(f'{save_dir}/{main_type}_{sub_type}_{im_type}')
    generate_pbr_dir = merge(save_dir)
    return generate_pbr_dir


def mask_hist(image_pth, mask_pth, pixel):
    """Histogram equalization (mask-level)"""
    image = cv2.imread(image_pth)
    image = cv2.resize(image, (pixel, pixel))
    mask = cv2.imread(mask_pth, 0)  
    mask = cv2.resize(mask, (pixel, pixel))

    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    masked_img = np.zeros_like(image)

    for i in range(3):  # RGB 3 channels
        channel = image[:, :, i]
        masked_channel = cv2.bitwise_and(channel, channel, mask=binary_mask)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_channel = clahe.apply(masked_channel)
        masked_img[:, :, i] = cv2.bitwise_or(clahe_channel, np.bitwise_and(channel, ~binary_mask))
    
    return masked_img


def normal_hist(img_pth, pixel):
    """Histogram equalization (image-level)"""
    img = cv2.imread(img_pth)
    img = cv2.resize(img, (pixel, pixel))
    b, g, r = cv2.split(img)
    equ_b = cv2.equalizeHist(b)
    equ_g = cv2.equalizeHist(g)
    equ_r = cv2.equalizeHist(r)
    equ_img = cv2.merge((equ_b, equ_g, equ_r))
    return equ_img


def convert_cv2_to_pil(opencv_img):
    """Convert cv2 image format to pil"""
    color_coverted = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(color_coverted)
    return pil_img


def divide_masks(mask_dir='experiments/obj_train/masks'):
    """Refine the intersecting parts of the generated masks"""
    mask_pth_list = [os.path.join(mask_dir, name) for name in os.listdir(mask_dir)]

    bi_masks = []
    true_counts = []

    for mask_pth in mask_pth_list:
        mask = Image.open(mask_pth)
        mask = (np.array(mask) == 255)
        bi_masks.append(mask)
        true_counts.append(np.sum(mask))

    sorted_indices = np.argsort(true_counts)
    sorted_mask_pth = [mask_pth_list[i] for i in sorted_indices]
    sorted_bi_masks = [bi_masks[i] for i in sorted_indices]

    for i in range(len(sorted_bi_masks)):
        for j in range(i+1, len(sorted_bi_masks)):
            mask_i = sorted_bi_masks[i]
            mask_j = sorted_bi_masks[j]
            # caculate intersection
            intersection = np.logical_and(mask_i, mask_j)
            # set intersection with smaller areas to false
            mask_i[intersection] = False
            sorted_bi_masks[i] = mask_i

        mask = sorted_bi_masks[i] * 255 
        mask_im = Image.fromarray(mask.astype('uint8'),'L')
        mask_im.save(sorted_mask_pth[i])

def find_closest_key(albedo_avgs, number):
    """Find the index of the closest neighbor block by albedo"""
    closest_key = None
    min_diff = float('inf')
    
    for key, value in albedo_avgs.items():
        diff = abs(value - number)
        if diff < min_diff:
            closest_key = key
            min_diff = diff

    return closest_key

def merge(mat_parts_dir):
    """Merge different material maps into whole texture maps."""
    pbr_list1 = ['roughness', 'metallic', 'diffuse', 'specular', 'basecolor', 'normal']
    pbr_list2 = ['height', 'displacement']

    mat_whole_dir = mat_parts_dir.replace('mat_parts_pbr', 'mat_whole_pbr')
    os.makedirs(mat_whole_dir, exist_ok=True)

    for mat_pbr in pbr_list1:
        mat_im_list = [os.path.join(mat_parts_dir, name) for name in os.listdir(mat_parts_dir) if mat_pbr in name]

        if mat_im_list != []:
            first_image = Image.open(mat_im_list[0])
            first_image_np = np.array(first_image)
            accumulated_image = np.zeros_like(first_image_np)

        for im_pth in mat_im_list:
            mat_image = Image.open(im_pth)
            image_np = np.array(mat_image)

            accumulated_image = accumulated_image + image_np

        accumulated_image = np.clip(accumulated_image, 0, 255)
        accumulated_image_pil = Image.fromarray(accumulated_image.astype(np.uint8))
        accumulated_image_pil.save(f"{mat_whole_dir}/{mat_pbr}.png")
    
    for mat_pbr in pbr_list2:
        mat_im_list = [os.path.join(mat_parts_dir, name) for name in os.listdir(mat_parts_dir) if mat_pbr in name]

        if mat_im_list != []:
            first_image = Image.open(mat_im_list[0])
            first_image_np = np.array(first_image)
            accumulated_image = np.zeros_like(first_image_np)

        for im_pth in mat_im_list:
            mat_image = Image.open(im_pth)
            image_np = np.array(mat_image.convert('I'))

            accumulated_image = accumulated_image + image_np

        accumulated_image_pil = Image.fromarray(accumulated_image.astype(np.uint32))
        accumulated_image_pil.save(f"{mat_whole_dir}/{mat_pbr}.png")
    return mat_whole_dir

def format_string(s):
    """
    Formats a given string to help find correct name from material lib.
    
    Args:
    s (str): The input string to be formatted.

    Returns:
    str: The formatted string.

    Example:
    >>> input_str = 'MetalPlates015A'
    >>> formatted_str = format_string(input_str)
    >>> print(formatted_str)
    'acg_metal_plates_015_a'
    """
    s = re.sub(r'(?<!^)(?=[A-Z])', '_', s)
    s = re.sub(r'(\d)([A-Za-z])', r'\1_\2', s)
    s = re.sub(r'([A-Za-z])(\d)', r'\1_\2', s)
    return 'acg_' + s.lower()
