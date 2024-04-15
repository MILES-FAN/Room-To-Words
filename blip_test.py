import cv2
import os
import torch
from wd_tagger import wd_tagger
from lavis.models import load_model_and_preprocess
from PIL import Image

device = torch.device("cuda")if torch.cuda.is_available() else torch.device("cpu")
if torch.cuda.is_available():
    print("Using GPU")
else:
    print("Using CPU")

blip2_model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_opt", model_type="pretrain_opt2.7b", is_eval=True, device=device)

def caption_image(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img_rgb)
    processed_frame = vis_processors["eval"](img).unsqueeze(0).to(device)
    caption = blip2_model.generate({"image": processed_frame})
    return caption

def tag_image(img):
    tags = wd_tagger().tag_image(img)
    return tags

def process(img_name, enable_tags=True):
    if img_name.endswith('.jpg'):
        img = cv2.imread(f'{img_name}')
        caption = caption_image(img)
        tags = ""
        if enable_tags:
            tags = tag_image(img_name)
        print(f'Image: {img_name}')
        print(f'Caption: {caption}')
        print(f'Tags: {tags}')
        with open(f'{img_name}.txt', 'a') as f:
            f.write(f'Caption: {caption}\n')
            if enable_tags:
                f.write(f'Tags: {tags}\n')


