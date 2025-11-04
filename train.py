
from __future__ import print_function
from sklearn.model_selection import train_test_split
import keras
import os
import numpy as np
from utils import ADAMLearningRateTracker
import SCIENetModel
from keras.optimizers import adam_v2
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from generators import mybatch_generator_train_bi_unet, mybatch_generator_validation_bi_unet
import pandas as pd
from utils import get_input_image_names_bi
from keras.utils.vis_utils import plot_model
from keras import losses,metrics

from keras import backend as K


os.environ['CUDA_VISIBLE_DEVICES']='0'

import tensorflow as tf
from tensorflow.keras.losses import Loss

class WeightMAE():
    def call(self, y_true, y_pred):
        abs_y_true = tf.abs(y_true)
        clipped_y_true = tf.clip_by_value(abs_y_true,0,0.5)
        weight = 1 -  clipped_y_true
        abs_error = tf.abs(y_true - y_pred)
        weight_abs_error = weight * abs_error
        weight_mae = tf.reduce_mean(weight_abs_error)
        return weight_mae
        

def weighted_mae(y_true, y_pred):
    abs_y_true = K.abs(y_true)
    clipped_y_true = K.clip(abs_y_true,0,0.5)
    weight = 1 -  clipped_y_true
    abs_error = K.abs(y_true - y_pred)
    weight_abs_error = weight * abs_error
    weight_mae = K.mean(weight_abs_error)
    return weight_mae


def train():
    model = SCIENetModel.model_arch_cloud(input_rows=in_rows,
                                       input_cols=in_cols,
                                       num_of_channels=num_of_channels,
                                       num_of_classes=num_of_classes)
    model.compile(optimizer=adam_v2.Adam(lr=starting_learning_rate), loss=weighted_mae, metrics=[metrics.mae])
    model.summary()

    model_checkpoint = ModelCheckpoint(weights_path, monitor='val_mean_absolute_error', save_best_only=True)
    lr_reducer = ReduceLROnPlateau(factor=decay_factor, cooldown=0, patience=patience, min_lr=end_learning_rate, verbose=1)
    csv_logger = CSVLogger(experiment_name + '_log_1.log')


    train_img_split = train_img
    train_msk_split =train_msk
    val_img_split = val_img
    val_msk_split = val_msk
    
    if train_resume:
        model.load_weights(weights_path)
        print("\nTraining resumed...")
    else:
        print("\nTraining started from scratch... ")

    print("Experiment name: ", experiment_name)
    print("Input image size: ", (in_rows, in_cols))
    print("Number of input spectral bands: ", num_of_channels)
    print("Learning rate: ", starting_learning_rate)
    print("Batch size: ", batch_sz, "\n")
    print(mybatch_generator_train_bi_unet(list(zip(train_img_split, train_msk_split)), in_rows, in_cols, batch_sz, max_bit))

    history = model.fit_generator(
        generator=mybatch_generator_train_bi_unet(list(zip(train_img_split, train_msk_split)), in_rows, in_cols, batch_sz, max_bit),
        steps_per_epoch=np.ceil(len(train_img_split) / batch_sz), epochs=max_num_epochs, verbose=1,
        validation_data=mybatch_generator_validation_bi_unet(list(zip(val_img_split, val_msk_split)), in_rows, in_cols, batch_sz, max_bit),
        validation_steps=np.ceil(len(val_img_split) / batch_sz),
        callbacks=[model_checkpoint, lr_reducer, ADAMLearningRateTracker(end_learning_rate),csv_logger])


GLOBAL_PATH = 'CloudL8-LC'
TRAIN_FOLDER = os.path.join(GLOBAL_PATH, 'TrainData')
VAL_FOLDER = os.path.join(GLOBAL_PATH, 'ValData')
TEST_FOLDER = os.path.join(GLOBAL_PATH, 'TestData')

in_rows = 512
in_cols = 512
num_of_channels = 6
num_of_classes = 3
starting_learning_rate = 1e-4
end_learning_rate = 1e-8
max_num_epochs = 2000  # just a huge number. The actual training should not be limited by this value
val_ratio = 0.2
patience = 15
decay_factor = 0.7
batch_sz = 4
max_bit = 1  # maximum gray level in landsat 8 images
experiment_name = "Pretrained_SCIENet"
weights_path = os.path.join(experiment_name + '.h5')
train_resume = False

# getting input images names

train_img, train_msk = get_input_image_names_bi(TRAIN_FOLDER, if_train=True)
val_img, val_msk = get_input_image_names_bi(VAL_FOLDER, if_train=True)

train()



