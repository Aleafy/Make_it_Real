from utils.gen_config import *
import sys 
sys.path.extend(['./som', './scripts/kaolin_scripts'])
from som.sesam_serial import seg_serial
import subprocess

from utils.gpt4_query import query_overall, query_refine, query_description, query_shape
from utils.mask_postprocess import refine_masks
from utils.texture_postprocess import region_refine, pixel_estimate
from utils.tools import *

from scripts.kaolin_scripts.load_cfg import render_model, paint_model_w_mask

import os
import argparse

def args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--obj_dir", default='data/mesh/gold_gun', type=str, help="Directory of .obj file.")
    parser.add_argument("--exp_name", default='gold_gun', type=str, help="Experiment name(unique)")
    parser.add_argument("--api_key", default='', type=str, help="Your own GPT-4V api key.") 
    parser.add_argument("--fine_grain", default=False, type=bool, help="Segmentation grain.")
    parser.add_argument("--view_num", default=1, type=int, help="Number of view points.")
    parser.add_argument("--leave_list", default=[], type=list, help="Viewpoints to be queried.")
    argv = parser.parse_args()
    return argv

if __name__ == '__main__':
    argv = args()
    obj_dir = argv.obj_dir 
    exp_name = argv.exp_name  
    obj_pth, texture_pth = get_texture_pth(obj_dir)
    texture_im, pixel_t = get_texture_img(texture_pth)
    
    folder_path = f'experiments/{exp_name}'
    result_path = f'experiments/{exp_name}/results'
    
    # Render albedo-only model from multi-view
    eval_cfg = gen_eval_yaml(exp_name, texture_pth, pixel_t, obj_pth)        
    render_model(eval_cfg)
    
    # segment viewpoint image to get initial semantic masks
    seg_serial(folder_path, argv.fine_grain) 
    # refine masks and mark different parts by numbers
    refine_masks(folder_path) 

    # OpenAI API Key
    api_key = argv.api_key
    leave_index = get_max_area(result_path)
    if argv.view_num > 1:
        if len(argv.leave_list) > 0: leave_list = argv.leave_list
        else:
            if leave_index < 8: leave_list = [leave_index, 7-leave_index]
            else: leave_list = [leave_index, 0]
    else:
        leave_list = [leave_index]

    mv_img_pth = paste_image(result_path)
    obj_info = query_shape(folder_path, mv_img_pth, api_key)

    # GPT-4V query about material of each part
    query_overall(folder_path, leave_list, api_key, obj_info)
    query_refine(folder_path, leave_list, api_key, obj_info)
    cache_path = query_description(folder_path, leave_list, api_key, obj_info, force=True)

    # sort gpt result
    most_result_pth, _ = sort_gpt_result(cache_path, leave_list)
    mat2im_path = sort_categories(most_result_pth)

    # Paint model based on masked image
    # get masked part(from 2D space to texture space)
    train_cfg = gen_train_yaml(exp_name, mat2im_path, result_path, pixel_t, obj_pth)
    paint_model_w_mask(train_cfg)

    masks_dir = os.path.join(folder_path.replace(exp_name, f'{exp_name}_train'), 'masks') 
    masks_refine_dir = region_refine(masks_dir, texture_pth, pixel=pixel_t) # region-level texture partition
    generate_pbr_dir = pixel_estimate(masks_refine_dir, texture_pth, pixel=pixel_t) # pixel-level brdf estimation

    # Render output
    # blender: render model with pbr maps
    output_refine_dir = f'output/refine_output/{exp_name}'
    command = 'blender'
    args = ['scripts/blender_scripts/paste_pbr.blend', '--background', "--python", "scripts/blender_scripts/render_pbr.py", "--", "--input_path", obj_pth, '--texture_path', generate_pbr_dir, '--output_path', output_refine_dir]
    process = subprocess.run([command] + args, check=True)

    # blender: render original model 
    output_ori_dir = f'output/ori_output/{exp_name}'
    command = 'blender'
    args = ['-b', "-P", "scripts/blender_scripts/render_ori.py", "--", "--input_path", obj_pth, '--output_path', output_ori_dir]
    subprocess.run([command] + args, check=True)