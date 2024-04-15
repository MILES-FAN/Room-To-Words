import os
import cv2
import numpy as np

def is_lowentropy_images(img, threshold=5):
    downsampled_img = cv2.resize(img, (256, 256))
    gray_img = cv2.cvtColor(downsampled_img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
    hist = hist / hist.sum()
    entropy = -1 * (hist * np.ma.log2(hist)).sum()
    #print(f'Entropy: {entropy}')
    return entropy < threshold

work_dir = r'C:\CODES\toolbox\item_matching_pipeline\test\hyb'

for root, dirs, files in os.walk(work_dir):
    for file in files:
        if file.endswith('.jpg') or file.endswith('.jpeg'):
            img = cv2.imread(f'{root}/{file}')
            #print(f'Processing {file}')
            if is_lowentropy_images(img):
                print(f'Low entropy image: {root}/{file}')