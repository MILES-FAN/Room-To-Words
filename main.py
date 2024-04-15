import cv2
from ultralytics import YOLO
import prepare_ds
import seg_test
import blip_test
import comparison
import image_generator
import os
import time

if 'LLM_API_KEY' not in os.environ and os.path.exists('add_env.py'):
    exec(open('add_env.py').read())

def process_data(data_dir):
    prepare_ds.prepare(data_dir, sub_img_size=1536, lod_sub_img_size=1024, overlap=2)
    cropped_objects = []
    start_time = time.time()
    print(os.listdir(data_dir))
    for dir in os.listdir(data_dir):
        if os.path.isdir(f'{data_dir}/{dir}'):
            print(f'Processing {dir}')
            cropped_objects.extend(seg_test.seg_dir(f'{data_dir}/{dir}'))
    
    cropped_objects = seg_test.remove_duplicates(cropped_objects)

    for i, cropped_object in enumerate(cropped_objects):
        file_name = cropped_object[3].split('/')[-1].split('.')[0]
        sub_dir = cropped_object[4]

        if not os.path.exists(f'{sub_dir}/{file_name}/items'):
            os.makedirs(f'{sub_dir}/{file_name}/items')
        cv2.imwrite(f'{sub_dir}/{file_name}/items/cropped_{i}.jpg', cropped_object[0])
        cv2.imwrite(f'{sub_dir}/{file_name}/items/bounding_box_{i}.jpg', cropped_object[1])
        with open(f'{sub_dir}/{file_name}/items/bounding_box_{i}.jpg.txt', 'w') as f:
            f.write(f'{cropped_object[2]}\n')
        with open(f'{sub_dir}/{file_name}/items/cropped_{i}.jpg.txt', 'w') as f:
            f.write(f'{cropped_object[2]}\n')

    crop_finished = time.time()

    for dir in os.listdir(data_dir):
        if os.path.isdir(f'{data_dir}/{dir}'):
            for file in os.listdir(f'{data_dir}/{dir}'):
                if file.endswith('.jpg'):
                    blip_test.process(f'{data_dir}/{dir}/{file}')
    
    for file in os.listdir(data_dir):
        if file.endswith('.jpg'):
            blip_test.process(f'{data_dir}/{file}') 
    
    end_time = time.time()
    print(f'Processing completed in {end_time - start_time} seconds.\nCrop time: {crop_finished - start_time} seconds\nBlip time: {end_time - crop_finished} seconds.')

def main():
    work_dir = './rooms'
    for dir in os.listdir(work_dir):
        if os.path.isdir(f'{work_dir}/{dir}'):
            process_data(f'{work_dir}/{dir}')
    prompt_for_generate = image_generator.generate_prompt(comparison.make_comparison(work_dir))
    image_url = image_generator.generate_image(prompt_for_generate)
    print(f'Generated image: {image_url}')
if __name__ == '__main__':
    main()