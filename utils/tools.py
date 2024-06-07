import json
import os

from PIL import Image
import re

import numpy as np

def sort_gpt_result(input_pth, leave_list):
    """Organize the GPT query results into a dictionary with keys as <img_num>_<part_num> and values as material names."""
    result = json.load(open(input_pth,'r'))
    des_dic = {}
    for file_name, ann in result.items():
        leave_number = int(file_name.split('/')[-2])
        if leave_number in leave_list:
            try:
                mat_dic = ann["choices"][0]["message"]["content"]
                img_name = '_'.join(file_name.split('_')[:-1])
                part_number = file_name.split('_')[-1]

                mat_dic = pp_dic_result(mat_dic)
                if mat_dic is None: continue

                des_dic[f"{img_name}, {int(part_number)}"] = list(mat_dic.keys())
            except:
                print(f"error {file_name}")
                continue
    
    result_pth = os.path.join(os.path.split(input_pth)[0], 'sort_result.json')
    json.dump(des_dic, open(result_pth,'w'), indent=1)
    most_des_dic, diff_count = select_most_frequent_materials(des_dic)
    most_result_pth = os.path.join(os.path.split(input_pth)[0], 'sort_most_result.json')
    json.dump(most_des_dic, open(most_result_pth,'w'), indent=1)

    return most_result_pth, diff_count

def count_materials(materials_data):
    """Count the occurrences of each material."""
    material_count = {}

    for materials in materials_data.values():
        for material in materials:
            if material in material_count:
                material_count[material] += 1
            else:
                material_count[material] = 1

    return material_count

def select_most_frequent_materials(materials_data):
    "Vote to select the most frequent material in the current part, applicable for top-k queries (for k > 1)."
    material_count = count_materials(materials_data)
    most_frequent_materials = {}
    
    for image, materials in materials_data.items():
        materials_sorted = sorted(materials, key=lambda x: material_count[x], reverse=True)
        most_frequent_material = materials_sorted[0]
        most_frequent_materials[image] = most_frequent_material
    
    material_name = []
    for mat_name in most_frequent_materials.values():
        if mat_name not in material_name:
            material_name.append(mat_name)

    return most_frequent_materials, len(material_name)

def paste_image(result_pth, leave_list=[0,3,4,7]):
    "Paste multiple images together."
    re_images = []
    pixel = 800
    for root, _, files in os.walk(result_pth):
        for f in files:
            number = int(f.split('_')[2])
            if number not in leave_list: continue
            im_path = os.path.join(root, f)

            image = Image.open(im_path)
            re_image = image.resize((pixel,pixel), Image.BICUBIC)
            re_images.append(re_image)

    new_image = Image.new('RGB', (pixel*4, pixel))
    for i, img in enumerate(re_images):
        new_image.paste(img, (i*pixel, 0))
    
    mv_img_pth = os.path.join(os.path.split(result_pth)[0], 'mv_image.png')
    new_image.save(mv_img_pth)
    print('paste done!')
    return mv_img_pth


def sort_categories(im2mat_pth):
    """Organize the dictionary results into the {material: part} format."""
    im_2mat = json.load(open(im2mat_pth, 'r'))
    result_pth = im2mat_pth.replace('sort_most_result', 'result_mat2im')
    mat_2im = {}
    for im, mat in im_2mat.items():
        if mat not in mat_2im.keys():
            mat_2im[mat] = []
        
        mat_2im[mat].append(im)
    
    json.dump(mat_2im, open(result_pth, 'w'), indent=1)
    return result_pth

def get_texture_pth(obj_dir):
    """
    obj_dir: include .mtl, .obj, .png texture files 
    return: obj_pth, texture_pth
    """

    # Initialize variables to store paths
    obj_pth = None
    mtl_pth = None
    texture_pth = None

    # Search for .obj and .mtl files
    for root, _, files in os.walk(obj_dir):
        for file in files:
            if file.endswith('.obj'):
                obj_pth = os.path.relpath(os.path.join(root, file), obj_dir)
            elif file.endswith('.mtl'):
                mtl_pth = os.path.relpath(os.path.join(root, file), obj_dir)

    # Read the .mtl file and find the line starting with map_Kd
    if mtl_pth:
        with open(os.path.join(obj_dir, mtl_pth), 'r') as mtl_file:
            lines = mtl_file.readlines()
            for line in lines:
                if line.startswith('map_Kd'):
                    texture_pth = line.split()[-1]  # Get the texture path

    return os.path.join(obj_dir, obj_pth), os.path.join(obj_dir, texture_pth)



def calculate_non_white_area(image_path):
    """Calculate the area of the non-white regions in the given image."""
    image = Image.open(image_path)
    image = image.convert("RGBA")
    
    non_white_pixels = 0
    for pixel in image.getdata():
        if pixel[0] < 250 or pixel[1] < 250 or pixel[2] < 250: 
            non_white_pixels += 1
            
    return non_white_pixels

def get_max_area(result_path):
    """Calculate the area of the non-white regions in all images within the specified directory and return a list of these areas."""
    areas = []
    
    for filename in os.listdir(result_path):
        if filename.endswith('_rgb.jpg'):
            full_path = os.path.join(result_path, filename)
            area = calculate_non_white_area(full_path)
            if '0008' in filename or '0009' in filename:
                area = 0.85*area
            areas.append(area)
    max_area = max(areas)
    max_index = areas.index(max_area)
    return max_index

def pp_dic_result(output_str):
    """Post-process the dict format result generated by GPT-4."""
    try:
        dict_pattern = r"\{.*?\}"
        dict_strs = re.findall(dict_pattern, output_str)
        if 'json' in output_str:
            output_str = output_str[output_str.index('{'):output_str.index('}')+1]
            combined_dict = eval(output_str)
        elif 'sorry' in output_str: 
            print("waiting for rate limit...")
            return None
        elif len(dict_strs) > 1:
            combined_dict = {}
            for dict_str in dict_strs:
                dict_item = eval(dict_str)
                combined_dict.update(dict_item)
        else:
            combined_dict = eval(output_str)
        if type(combined_dict) == tuple:
            return combined_dict[0]
        return combined_dict
        
    except:
        return None
    

def check_mat_valid(mat, count):
    valid_mat = ['Blends', 'Ceramic', 'Concrete', 'Fabric', 'Ground', 'Leather', 'Marble', 'Metal', 'Plaster', 'Plastic', 'Stone', 'Terracotta', 'Wood', 'Misc']
    change_mat = {"Rubber": "Plastic", "Glass": "Metal", "Bone": "Marble"}
    if mat not in valid_mat:
        if mat in change_mat:
            return change_mat[mat]
        else:
            if count > 3: return 'Misc'
            else: return None
    else: return mat


def get_texture_img(texture_pth):
    """get texture image, PIL format, square-size"""
    texture_im = Image.open(texture_pth)
    pixel_t = min(np.array(texture_im).shape[0], np.array(texture_im).shape[1])
    if np.array(texture_im).shape[0] != np.array(texture_im).shape[1]:
        texture_im.resize((pixel_t, pixel_t)).save(texture_pth)
    return texture_im, pixel_t
    