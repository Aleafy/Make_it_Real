import yaml
import shutil

import os
import glob 


def gen_eval_yaml(input_str, texture_pth, pixel, obj_pth):
    # The original YAML content as a dictionary
    yaml_content = {
        "log": {
            "exp_name": f"{input_str}",
            "eval_only": True
        },
        "render": {
            "eval_grid_size": 1200
        },
        "guide": {
            # "text": f"A photo of {obj_name}, " + "{} view",
            # "append_direction": True,
            "shape_path": obj_pth, 
            "initial_texture": texture_pth, 
            "texture_resolution": pixel
        },
        "optim": {
            "seed": 3
        }
    }

    # Save the modified YAML content to a file
    yaml_file_path = 'configs/eval/render_mv_image.yaml'
    with open(yaml_file_path, 'w') as file:
        yaml.dump(yaml_content, file, sort_keys=False)
    return yaml_file_path


def gen_train_yaml(input_str, mat2im_path, result_path, pixel, obj_pth):
    # The original YAML content as a dictionary
    yaml_content = {
        "log": {
            "exp_name": f"{input_str}_train"
        },
        "render": {
            "eval_grid_size": 1200
        },
        "guide": {
            # "text": f"A photo of {input_str}, " + "{} view",
            # "append_direction": True,
            "shape_path": obj_pth, 
            "mat2im_path": mat2im_path,
            "result_path": result_path,
            "texture_resolution": pixel
        },
        "optim": {
            "seed": 3
        }
    }

    # Save the modified YAML content to a file
    yaml_file_path = 'configs/train/gen_mat_masks.yaml'
    with open(yaml_file_path, 'w') as file:
        yaml.dump(yaml_content, file, sort_keys=False)
    return yaml_file_path

def remove_file(folder_path, suffix):
    """Delete files with the specified suffix."""
    pt_files = glob.glob(os.path.join(folder_path, f'*{suffix}'))
    for file_path in pt_files:
        try:
            os.remove(file_path)
            print(f"Deleted file: {file_path}")
        except OSError as e:
            print(f"Error: {file_path} : {e.strerror}")

def remove_dir(folder_path):
    """Attempt to delete a folder and all its contents."""
    try:
        shutil.rmtree(folder_path)
        print(f"The directory {folder_path} has been removed successfully")
    except Exception as e:
        print(f"Error: {e}")