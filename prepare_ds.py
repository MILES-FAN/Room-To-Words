import cv2
from ultralytics import YOLO
import os
import numpy as np

def split_image(img, sub_img_size, overlap):
    sub_imgs = []
    horizontal_splits = int((img.shape[1] // sub_img_size) * overlap)
    vertical_splits = int((img.shape[0] // sub_img_size) * overlap)
    for i in range(horizontal_splits):
        for j in range(vertical_splits):
            x = int(i * img.shape[1] / horizontal_splits)
            y = int(j * img.shape[0] / vertical_splits)
            if x + sub_img_size > img.shape[1]:
                x = img.shape[1] - sub_img_size
            if y + sub_img_size > img.shape[0]:
                y = img.shape[0] - sub_img_size
            sub_imgs.append(img[y:y + sub_img_size, x:x + sub_img_size])
    return sub_imgs

def is_lowentropy_images(img, threshold=5):
    downsampled_img = cv2.resize(img, (256, 256))
    gray_img = cv2.cvtColor(downsampled_img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
    hist = hist / hist.sum()
    entropy = -1 * (hist * np.ma.log2(hist)).sum()
    print(f'Entropy: {entropy}')
    return entropy < threshold

def prepare(data_dir, sub_img_size=1536, lod_sub_img_size=1024, overlap=2, max_size=4096):
    # Load image
    imgs = []

    for file in os.listdir(data_dir):
        if file.endswith('.jpg') or file.endswith('.jpeg'):
            img = cv2.imread(f'{data_dir}/{file}')
            # downscale image if it is too large
            if img.shape[0] > max_size or img.shape[1] > max_size:
                scale = max_size / max(img.shape[0], img.shape[1])
                img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
            img_pair = (img, f'{data_dir}/{file}')
            imgs.append(img_pair)
    
    for img_pair in imgs:
        # Split image into sub-images
        img = img_pair[0]
        file_name = img_pair[1].split('/')[-1].split('.')[0]
        sub_imgs = split_image(img, sub_img_size, overlap)
        lod_sub_imgs = split_image(img, lod_sub_img_size, overlap)

        if not os.path.exists(f'{data_dir}/{file_name}'):
            os.makedirs(f'{data_dir}/{file_name}')

        for i,sub_img in enumerate(sub_imgs):
            img_name = f'{data_dir}/{file_name}/sub_{i}.jpg'
            if is_lowentropy_images(sub_img):
                continue
            cv2.imwrite(img_name, sub_img)

        for i, sub_img in enumerate(lod_sub_imgs):
            img_name = f'{data_dir}/{file_name}/lod_sub_{i}.jpg'
            if is_lowentropy_images(sub_img):
                continue
            cv2.imwrite(img_name, sub_img)

        cv2.imwrite('{data_dir}/{file_name}/full.jpg', img)
