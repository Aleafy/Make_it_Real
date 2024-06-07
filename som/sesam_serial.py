import torch
from PIL import Image

from semantic_sam.BaseModel import BaseModel
from semantic_sam import build_model
from semantic_sam.utils.arguments import load_opt_from_config_file
from som.task_adapter.semantic_sam.tasks import inference_semsam_auto

import os

'''
build args
'''
semsam_cfg = "som/configs/semantic_sam_only_sa-1b_swinL.yaml"
semsam_ckpt = "som/ckpts/swinl_only_sam_many2many.pth"
opt_semsam = load_opt_from_config_file(semsam_cfg)

'''
build model
'''
model_semsam = BaseModel(opt_semsam, build_model(opt_semsam)).from_pretrained(semsam_ckpt).eval().cuda()


@torch.no_grad()
def inference(image, slider, mode, alpha, label_mode, anno_mode,points_per_side,save_dir, *args, **kwargs):
    if slider < 1.5:
        model_name = 'seem'
    elif slider > 2.5:
        model_name = 'sam'
    else:
        if mode == 'Automatic':
            model_name = 'semantic-sam'
            if slider < 1.5 + 0.14:                
                level = [1]
            elif slider < 1.5 + 0.28:
                level = [2]
            elif slider < 1.5 + 0.42:
                level = [3]
            elif slider < 1.5 + 0.56:
                level = [4]
            elif slider < 1.5 + 0.70:
                level = [5]
            elif slider < 1.5 + 0.84:
                level = [6]
            else:
                level = [6, 1, 2, 3, 4, 5]
        else:
            model_name = 'sam'


    if label_mode == 'Alphabet':
        label_mode = 'a'
    else:
        label_mode = '1'

    text_size, hole_scale, island_scale=1200,100,100 
    text, text_part, text_thresh = '','','0.0'
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        semantic=False

        if model_name == 'semantic-sam':
            model = model_semsam
            output = inference_semsam_auto(model, image, level, text, text_part, text_thresh, text_size, hole_scale, island_scale, semantic, label_mode=label_mode, alpha=alpha, anno_mode=anno_mode, points_per_side=points_per_side, save_dir=save_dir, *args, **kwargs)

        return output
    

def seg_serial(input_dir, fine_grain=False):
    img_lst = []
    img_names = []
    seg_slider = 1.92
    if fine_grain:
        seg_slider = 2.35
    for root, _, files in os.walk(input_dir):
        for f in files:
            if 'rgb.jpg' not in f:
                continue
            f_name = os.path.join(root, f)
            img_names.append(f.split('.')[0])
            img_lst.append(f_name)

    for i in range(len(img_names)):
        image = Image.open(img_lst[i]).convert('RGB')
        output_dir = input_dir +'/ori_masks/'+ img_names[i]
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        inference(image, slider=seg_slider, mode='Automatic', alpha=0.1, label_mode='Number', anno_mode=['Mask', 'Mark'], points_per_side=32, save_dir=output_dir) 


def seg_serial_single(input_dir, img_index, seg_slider=1.92):
    #TODO change args
    img_pth = f'{input_dir}/results/step_00000_000{img_index}_rgb.jpg'
    image = Image.open(img_pth).convert('RGB')
    output_dir = input_dir +'/ori_masks/'+ f'step_00000_000{img_index}_rgb'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    inference(image, slider=seg_slider, mode='Automatic', alpha=0.1, label_mode='Number', anno_mode=['Mask', 'Mark'], points_per_side=32, save_dir=output_dir) 