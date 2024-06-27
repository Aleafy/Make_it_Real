import base64
import requests
import json
import os
from tqdm import tqdm
import time
import re

from openai import OpenAI

# Set proxy environment variables for HTTP and HTTPS requests, to ensure proper functioning of GPT-4 in certain regions.
# os.environ['http_proxy'] = 'http://<your_proxy>.example.com/'
# os.environ['https_proxy'] = 'http://<your_proxy>.example.com/'
# os.environ['HTTP_PROXY'] = 'http://<your_proxy>.example.com/'
# os.environ['HTTPS_PROXY'] = 'http://<your_proxy>.example.com/'

prompt = """You are showed with a group of spherical PBR materials(all made of {}, which is a kind of {}, in total {}), can you generate a caption for each(about 20~30 words), try to distinguish their difference. Only describe the appearance features (must including `color` and detailed `material`(such as patterns, roughness, metalness, concave and convex patterns, condition)), and don't give too much other information. Do not describe whether it is reflective or not. Do not describe the shape of overall object(such as sphere). Please use a dictionary to represent the output result {}"""

# Your own OpenAI API Key
api_key = "" 
assert api_key, "Error: api_key is empty! Please replace with Your own api_key."

def process_image(image_path):
    # Function to encode the image
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    proc_pth = image_path.split('/')[-1].replace('.jpg','')
    img_num = proc_pth.split('_')[-1]
    cat_name = proc_pth.split('_')[0]
    sub_name = proc_pth.split('_')[2]
    match = re.match(r"^[^\d]+", sub_name)
    sub_name = match.group()
    des = ""
    for i in range(int(img_num)):
        des += f"{i+1}: Description {i+1}"
        if i < int(img_num)-1:
            des += ", "
    des = "{" + des + "}"
    new_prompt = prompt.format(sub_name, cat_name, img_num, des)
    print(new_prompt)

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
                "text": new_prompt
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

    api_base = "https://api.openai.com/v1/chat/completions" # you can switch to another accessible GPT-4 API host
    # response = requests.post(api_base, headers=headers, json=payload) # another way to get response

    client = OpenAI(api_key=api_key,
                base_url=api_base)
    response = client.chat.completions.create(**payload)
    return response

if __name__ == "__main__":
    if not os.path.exists("./cache_result.json"):
        json.dump({}, open("./cache_result.json", 'w'), indent=4)
    data = json.load(open("./cache_result.json"))
    
    for img_pth in tqdm(os.listdir("./images_combine_demo")):
        if img_pth in data.keys():
            continue        
        try:
            response = process_image("./images_combine_demo/"+img_pth)
        except:
            time.sleep(2)
            continue
        data[img_pth] = json.loads(response.json())
        json.dump(data, open("./cache_result.json", 'w'), indent=4)
        