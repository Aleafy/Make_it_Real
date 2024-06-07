import base64
import requests
import json
import os
from tqdm import tqdm
import time
from utils.tools import pp_dic_result, check_mat_valid

from openai import OpenAI

### 0. Query overall shape (optional)
### 1. Query maintype
### 2. Query subtype
### 3. Query best-matched description

add_prompt = """This is a(or a group of) 3D rendered image(s) of an object using a pure texture map without PBR material information. Please use its visual characteristics (including color, pattern, object type) along with your existing knowledge to infer the materials of different parts. """

valid_mat = ['Blends', 'Ceramic', 'Concrete', 'Fabric', 'Ground', 'Leather', 'Marble', 'Metal', 'Plaster', 'Plastic', 'Stone', 'Terracotta', 'Wood', 'Misc']

prompt_overall = add_prompt + """Identify the material of each part(marked with Arabic numerals), presented in the exact form of {number_id: \"material\"}. Don't output other information. (optional list of material is [Ceramic, Concrete, Fabric, Ground, Leather, Marble, Metal, Plaster, Plastic, Stone, Terracotta, Wood, Misc], The 'Misc' category is output when nothing else matches.) """

prompt_refine = """{}\n\nSelect the most similar {} material type of number {} part of the image, according to the analysis of corresponding part material(including color, pattern, roughness, age and so on...). If you find it difficult to subdivide, just output {}. Don't output other information. Only a single word representing the category from optional list needs to be output. (optional list of material is {}). """

prompt_refine_sub = """{}\n\nLook at the material carefully of number {} part of the image, here are some descriptions about {} materials, can you tell me which is the best description match the part {} in the image?\n{}
Just tell me the final result in dict format with material name and descrption. Don't output other information.
"""

prompt_shape = add_prompt + """This is an image from four perspectives of an object. Please tell me a description, including what kind of object it is, what parts it has, and what materials each part is made of. No more than 50 words."""

prompt_obj_append = "\nThis is a description about the object to help you understand the overall 3D object: {}"

def process_image(image_path, api_key, prompt):
    # Function to encode the image
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
        
    # Getting the base64 string
    base64_image = encode_image(image_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": prompt
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
                }
            ]
            }
        ],
        "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    return response


def query_shape(folder_path, img_pth, api_key, type_str=None):
    """Query overall information(e.g., shape, type) of 3D object.(help to query material)"""
    cache1_pth = f"{folder_path}/gpt4_query/shape_result.json"
    os.makedirs(os.path.split(cache1_pth)[0], exist_ok=True)
    if not os.path.exists(cache1_pth):
        json.dump({}, open(cache1_pth, 'w'), indent=4)
    data = json.load(open(cache1_pth))
    response, count, success = '', 0, False
    if img_pth not in data.keys():
        while count < 5:
            try:
                if type_str is not None: 
                    new_prompt_shape = f"{prompt_shape} (Object type: This is a {type_str})"
                else:
                    new_prompt_shape = prompt_shape
                response = process_image(img_pth, api_key, new_prompt_shape)
                if 'error' in json.loads(response.json()).keys():
                    print("waiting for rate limit...")
                    time.sleep(5)
                    continue
                print(f"success: query {img_pth}")
                success = True
                break
            except:
                count += 1
                print(f"error: query {img_pth}")
                time.sleep(5)
                continue
        if success:        
            data[img_pth] = json.loads(response.json())
            json.dump(data, open(cache1_pth, 'w'), indent=4)
    if count == 5: return None
    if data[img_pth] != "":
        shape_info = data[img_pth]["choices"][0]["message"]["content"]
        print(shape_info)
        return shape_info
    else:
        return None

def query_overall(folder_path, leave_list, api_key, obj_info=None):
    """Query main material type of different parts of the object."""
    cache1_pth = f"{folder_path}/gpt4_query/cache_result_1.json"
    os.makedirs(os.path.split(cache1_pth)[0], exist_ok=True)
    if not os.path.exists(cache1_pth):
        json.dump({}, open(cache1_pth, 'w'), indent=4)
    data = json.load(open(cache1_pth))
    
    for root, _, files in tqdm(os.walk(f"{folder_path}/clean_masks")): # query the image with marks
        for f in files:
            leave_number = int(root.split('/')[-1])
            img_pth = os.path.join(root, f)
            if (leave_number not in leave_list) or ('full' not in f) or (img_pth in data.keys()):
                continue
             
            loop, count = True, 0
            while loop:
                count += 1
                if count > 5: break
                # gpt4 ask main type
                try:
                    new_prompt = prompt_overall
                    if obj_info is not None:
                        new_prompt = new_prompt + prompt_obj_append.format(obj_info)
                    response = process_image(img_pth, api_key, new_prompt)
                    response_json = json.loads(response.json())
                    desc = pp_dic_result(response_json["choices"][0]["message"]["content"])

                    # check if the return value is valid.
                    if desc is not None:
                        loop = False
                        for k, mat in desc.items():
                            c_mat = check_mat_valid(mat, count)
                            if c_mat is not None: 
                                desc[k] = c_mat # valid main type
                            else:
                                loop = True # continue to loop, gpt4 re-ask
                                break
                        if not loop:
                            print(f"success stage1: query {img_pth}\n{desc}")
                        else: continue
                    else: continue
                except:
                    print(f"error stage1: query {img_pth}")
                    time.sleep(2)
                    loop = True
                    continue
                
            response_json["choices"][0]["message"]["content"] = str(desc)
            data[img_pth] = response_json 
            json.dump(data, open(cache1_pth, 'w'), indent=4)


def query_refine(folder_path, leave_list, api_key, obj_info=None, force=False):
    """Query sub material type of different parts of the object."""
    result = json.load(open(f"{folder_path}/gpt4_query/cache_result_1.json",'r')) # maintype results
    sub_lst = json.load(open("./data/material_lib/annotations/category_tree.json")) # category tree
    cache2_pth = f"{folder_path}/gpt4_query/cache_result_2.json" # subtype results (to be queried)
    if not os.path.exists(cache2_pth):
        json.dump({}, open(cache2_pth, 'w'), indent=4)
    data = json.load(open(cache2_pth))

    for img_name, ann in result.items():
        leave_number = int(img_name.split('/')[-2])
        if leave_number not in leave_list:
            continue
        part_dic = ann["choices"][0]["message"]["content"]
        part_dic = pp_dic_result(part_dic)
        if part_dic is None: continue
        i, count, part_num = 0, 0, len(part_dic)
        
        while i < part_num:
            count += 1
            if count > part_num + 4: break
            suffix = list(part_dic.keys())[i]
            main_type = part_dic[suffix]
            new_img_name = f"{img_name}_{suffix}"
            if f"{img_name}_{suffix}" in data.keys():
                i = i + 1
                continue
            if main_type not in sub_lst:
                main_type = 'Misc'
            sub_type = sub_lst[main_type] # subtypes to be select from
            new_prompt_refine = prompt_refine.format(add_prompt, main_type, suffix, main_type, sub_type)
            if obj_info is not None:
                new_prompt_refine = new_prompt_refine + prompt_obj_append.format(obj_info)
            # gpt4 ask sub-type
            try:
                response = process_image(img_name, api_key, new_prompt_refine)
                response_json = json.loads(response.json())
                print(f"success stege2: query {img_name}_{suffix}")
            except:
                print(f"error stege2: query {img_name}_{suffix}")
                time.sleep(2)
                continue
            # check the value
            try:
                desc = response_json["choices"][0]["message"]["content"]
                if desc in sub_lst[main_type]: i = i + 1
                elif desc == 'Stone':
                    response_json["choices"][0]["message"]["content"] = 'PavingStones'
                    i = i + 1
                else: continue
            except: continue
            data[new_img_name] = response_json
            json.dump(data, open(cache2_pth, 'w'), indent=4)

def query_description(folder_path, leave_list, api_key, obj_info=None, force=False):
    """Query matched material description of different parts of the object."""
    result_1 = json.load(open(f"{folder_path}/gpt4_query/cache_result_1.json",'r')) # maintype info
    result_2 = json.load(open(f"{folder_path}/gpt4_query/cache_result_2.json",'r')) # subtype info
    sub_des = json.load(open("./data/material_lib/annotations/gpt_descriptions.json")) # highly-detailed annotation

    cache3_pth = f"{folder_path}/gpt4_query/cache_result_3.json" # matched description(to be queried)

    if not os.path.exists(cache3_pth):
        json.dump({}, open(cache3_pth, 'w'), indent=4)
    data = json.load(open(cache3_pth))

    for img_name, ann in tqdm(result_1.items()):
        leave_number = int(img_name.split('/')[-2])
        if leave_number not in leave_list:
            continue
        part_dic = ann["choices"][0]["message"]["content"]
        part_dic = pp_dic_result(part_dic)
        if part_dic is None: continue
            
        part_num, count, i = len(part_dic), 0, 0
        while i < part_num:
            suffix = list(part_dic.keys())[i]
            count += 1
            if count > part_num +4: break
            main_type = part_dic[suffix]
            new_img_name = img_name+f'_{suffix}'
            if new_img_name in data.keys():
                i += 1
                continue
            # gpt4 ask specific description
            try:
                if type(result_2[new_img_name]) == dict: 
                    sub_type = result_2[new_img_name]["choices"][0]["message"]["content"]
                else: # read cache directly
                    sub_type = result_2[new_img_name]
                description = sub_des[main_type+'_'+sub_type] 
                new_prompt = prompt_refine_sub.format(add_prompt, suffix, main_type, suffix, description)
                if obj_info is not None:
                    new_prompt = new_prompt + prompt_obj_append.format(obj_info)
                response = process_image(img_name, api_key, new_prompt)
                print(f"success stege3: query {img_name}_{suffix}")
            except:
                print(f"error stege3: query {img_name}_{suffix}")
                time.sleep(2)
                continue
            
            # check the value
            try:
                desc = pp_dic_result(json.loads(response.json())["choices"][0]["message"]["content"])
                if desc is not None: i+=1
                else: continue
            except: continue
            data[new_img_name] = json.loads(response.json())
            json.dump(data, open(cache3_pth, 'w'), indent=4)

    return cache3_pth