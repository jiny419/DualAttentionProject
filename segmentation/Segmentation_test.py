import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import glob
import os
from my_keras_applications.unet import *

epochs = 10
batch_size = 1
input_size, model = get_unet_128()
model.load_weights('weights/model_1.90-0.99.hdf5')

print(input_size)


test_filenames = glob.glob("input/test/*.jpg")

test_filenames = [filename.replace('\\','/').replace('.jpg', '') for filename in test_filenames]
test_filenames = [filename.split('/')[-1] for filename in test_filenames]




print('Predicting on {} samples with batch_size = {}...'.format(len(test_filenames), batch_size))
for start in tqdm(range(0, len(test_filenames), batch_size)):
    x_batch = []
    end = min(start + batch_size, len(test_filenames))
    ids_test_batch = test_filenames[start:end]
    for id in ids_test_batch:
        img = cv2.imread('input/test/{}.jpg'.format(id))
        x_batch.append(img)
    x_batch = np.array(x_batch, np.float32) / 255
    preds = model.predict_on_batch(x_batch)
    preds = np.squeeze(preds, axis=3)
    for index, pred in enumerate(preds):
        prob = np.array(pred).astype(np.float32) * 255
        current_filename = ids_test_batch[index]
        cv2.imwrite('input/test/segmentation/{}.png'.format(id), prob)
        

print("Done!")

