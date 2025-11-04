
from __future__ import print_function
import os
import numpy as np
import SCIENetModel
from generators import mybatch_generator_prediction_bi_unet
import tifffile as tiff
import pandas as pd
from utils import get_input_image_names_bi
from tensorflow.compat.v1.keras.models import model_from_json

import tensorflow as tf
import keras
import rasterio

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# keras.backend.tensorflow_backend.set_session(tf.Session(config=config))
os.environ['CUDA_VISIBLE_DEVICES']='0'

def prediction():
    model = SCIENetModel.model_arch_cloud(input_rows=in_rows,
                                       input_cols=in_cols,
                                       num_of_channels=num_of_channels,
                                       num_of_classes=num_of_classes)
    model.load_weights(weights_path, by_name=True)

    print("\nExperiment name: ", experiment_name)
    print("Prediction started... ")
    print("Input image size = ", (in_rows, in_cols))
    print("Number of input spectral bands = ", num_of_channels)
    print("Batch size = ", batch_sz)

    imgs_mask_test = model.predict_generator(
        generator=mybatch_generator_prediction_bi_unet(test_img, in_rows, in_cols, batch_sz, max_bit),
        steps=np.ceil(len(test_img) / batch_sz))

   # imgs_mask_test1 = np.argmax(imgs_mask_test, axis=-1)
   # imgs_mask_test1[imgs_mask_test1 ==2] = 255
   # imgs_mask_test1[imgs_mask_test1 ==1] = 128

    print("Saving predicted cloud masks on disk... \n")

    # pred_dir = experiment_name + '_train_256_test_256'
    pred_dir = experiment_name + '_Testdata'
    if not os.path.exists(os.path.join(pred_dir)):
        os.mkdir(os.path.join( pred_dir))

    for image, image_id in zip(imgs_mask_test, test_ids):
        with rasterio.open(os.path.join(TEST_FOLDER,'CloudyLandsat8',str(image_id))) as dst:
            profile = dst.profile
        img_saveresult = image.transpose(2,0,1)
        img_saveresult = img_saveresult.astype(np.float32)
        # image = (image[:, :, 0]).astype(np.float32)
        with rasterio.open(os.path.join(pred_dir, str(image_id)), mode = 'w', **profile) as dst:
            dst.write(img_saveresult)

     #   tiff.imsave(os.path.join( pred_dir, str(image_id)), image)


GLOBAL_PATH = 'CloudL8-LC'
#GLOBAL_PATH = '/media/tao/D1/NewPaper4/CloudL8-LC'
TRAIN_FOLDER = os.path.join(GLOBAL_PATH, 'TrainData')
VAL_FOLDER = os.path.join(GLOBAL_PATH, 'ValData')
TEST_FOLDER = os.path.join(GLOBAL_PATH, 'TestData')



in_rows = 512
in_cols = 512
num_of_channels = 6
num_of_classes = 3
batch_sz = 1
max_bit = 1  # maximum gray level in landsat 8 images
experiment_name = "Pretrained_SCIENet"
weights_path = os.path.join(experiment_name + '.h5')


# getting input images names

test_img, test_ids = get_input_image_names_bi(TEST_FOLDER, if_train=False)


prediction()
