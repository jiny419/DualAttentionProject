
import cv2
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard, LearningRateScheduler
from sklearn.model_selection import train_test_split
import keras.backend.tensorflow_backend as K
import glob
import os
from my_keras_applications.unet import *

batch_size = 2
_, model = get_unet_128()
#model.load_weights(filepath='weights/best_weights.hdf5') # For resuming train

train_img_path_template = 'input/train/{}.jpg'
train_img_mask_path_template = 'input/train/segmentation/{}.jpg'

train_filenames = glob.glob("input/train/*.jpg")
train_filenames = [filename.replace('\\','/').replace('.jpg', '') for filename in train_filenames]
train_filenames = [filename.split('/')[-1] for filename in train_filenames]

train_split, valid_split = train_test_split(train_filenames, test_size=0.10, random_state=42)

print('Training on {} samples'.format(len(train_split)))
print('Validating on {} samples'.format(len(valid_split)))



def train_generator():
    while True:
        train_split, valid_split = train_test_split(train_filenames, test_size=0.10, random_state=42)

        for start in range(0, len(train_split), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(train_split))
            ids_train_batch = train_split[start:end]
            for id in ids_train_batch:
                img  = cv2.imread(train_img_path_template.format(id))
                mask = cv2.imread(train_img_mask_path_template.format(id), cv2.IMREAD_GRAYSCALE)
                mask = np.expand_dims(mask, axis=2)
                x_batch.append(img)
                y_batch.append(mask)
            x_batch = np.array(x_batch, np.float32) / 255
            y_batch = np.array(y_batch, np.float32) / 255
            yield x_batch, y_batch


def valid_generator():
    while True:
        train_split, valid_split = train_test_split(train_filenames, test_size=0.10, random_state=42)

        for start in range(0, len(valid_split), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(valid_split))
            ids_valid_batch = valid_split[start:end]
            for id in ids_valid_batch:
                img  = cv2.imread(train_img_path_template.format(id))
                mask = cv2.imread(train_img_mask_path_template.format(id), cv2.IMREAD_GRAYSCALE)
                mask = np.expand_dims(mask, axis=2)
                x_batch.append(img)
                y_batch.append(mask)
            x_batch = np.array(x_batch, np.float32) / 255
            y_batch = np.array(y_batch, np.float32) / 255
            yield x_batch, y_batch



def lr_scheduler(epoch, lr):
    decay_rate = 0.9
    decay_step = 10
    if epoch % decay_step == 0 and epoch:
        return lr * decay_rate
    return lr

callbacks = [
             ModelCheckpoint(monitor='val_dice_loss',
                             #monitor='val_loss',
                             filepath='weights/' + model.name + '.{epoch:02d}-{val_dice_loss:.2f}.hdf5',
                             verbose = 1,
                             save_best_only=True,
                             save_weights_only=True,
                             mode='max'),
             TensorBoard(log_dir='logs'),
             LearningRateScheduler(lr_scheduler, verbose=1)]


model.fit_generator(generator=train_generator(),
                    steps_per_epoch=np.ceil(float(len(train_split)) / float(batch_size)),
                    epochs=100,  # 100회
                    verbose=1,
                    callbacks=callbacks,
                    validation_data=valid_generator(),
                    validation_steps=np.ceil(float(len(valid_split)) / float(batch_size)))
                    
