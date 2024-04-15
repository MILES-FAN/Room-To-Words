import cv2
import os
import numpy as np
import imagehash
from ultralytics import YOLO
from PIL import Image

model = YOLO('yolov8m-seg.pt')

def remove_duplicates(img_label_pair_list):
    hash_dict = {}
    for img_label_pair in img_label_pair_list:
        hash = imagehash.average_hash(Image.fromarray(img_label_pair[1]), hash_size=4)
        if hash not in hash_dict:
            hash_dict[hash] = img_label_pair
    return hash_dict.values()

def extract_objects(img_name, dir_name):
    img = cv2.imread(img_name)
    results = model(img_name)
    item_cnt = 0
    cropped_objects = []

    for r in results:
        mask = r.masks  # Segmentation mask

        for ci,c in enumerate(r):
            label = c.names[c.boxes.cls.tolist().pop()]
            # Create binary mask
            b_mask = np.zeros(img.shape[:2], np.uint8)

            #  Extract contour result
            contour = c.masks.xy.pop()
            #  Changing the type
            contour = contour.astype(np.int32)
            #  Reshaping
            contour = contour.reshape(-1, 1, 2)

            _ = cv2.drawContours(b_mask,
                            [contour],
                            -1,
                            (255, 255, 255),
                            cv2.FILLED)
            
            mask3ch = cv2.cvtColor(b_mask, cv2.COLOR_GRAY2BGR)
            isolated = cv2.bitwise_and(mask3ch, img)
            #  Bounding box coordinates
            x1, y1, x2, y2 = c.boxes.xyxy.cpu().numpy().squeeze().astype(np.int32)
            # Crop image to object region
            iso_crop = isolated[y1:y2, x1:x2]

            box_crop = img[y1:y2, x1:x2]

            img_info = f"{item_cnt}_{label}_({x1},{y1})_({x2},{y2})"

            img_label_pair = (iso_crop, box_crop, img_info, img_name, dir_name)
            cropped_objects.append(img_label_pair)
            item_cnt += 1

    return cropped_objects

def seg_dir(dir_name):
    cropped_objects = []
    original_cropped_count = len(cropped_objects)

    for file in os.listdir(dir_name):
        if file.endswith('.jpg'):
            cropped_objects.extend(extract_objects(f'{dir_name}/{file}', dir_name))

    cropped_objects = remove_duplicates(cropped_objects)
    print(f"Removed {original_cropped_count - len(cropped_objects)} duplicates.")
    return cropped_objects