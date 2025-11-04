import random
from skimage.io import imread
from skimage.transform import resize
import numpy as np
import tifffile as tiff
from tensorflow.keras.utils import to_categorical


def mybatch_generator_train_bi_unet(zip_list, img_rows, img_cols, batch_size, shuffle=True, max_possible_input_value=1):
    number_of_batches = np.ceil(len(zip_list) / batch_size)
    if shuffle:
        random.shuffle(zip_list)
    counter = 0

    while True:
        if shuffle:
            random.shuffle(zip_list)

        batch_files = zip_list[batch_size * counter:batch_size * (counter + 1)]
        image_list = []
        mask_list = []
        mask2_list = []
        for file, mask in batch_files:

            image= tiff.imread(file[0])
            
            Landsat = tiff.imread(mask[1])
            Landsat = np.nan_to_num(Landsat)


            image = np.nan_to_num(image)

            # mask = mask[..., np.newaxis]

            image_list.append(image)
            mask2_list.append(Landsat)

        counter += 1
        image_list = np.array(image_list)
        mask2_list = np.array(mask2_list)
        yield (image_list, mask2_list)

        if counter == number_of_batches:
            if shuffle:
                random.shuffle(zip_list)
            counter = 0

def mybatch_generator_validation_bi_unet(zip_list, img_rows, img_cols, batch_size, shuffle=False, max_possible_input_value=1):
    number_of_batches = np.ceil(len(zip_list) / batch_size)
    if shuffle:
        random.shuffle(zip_list)
    counter = 0

    while True:

        batch_files = zip_list[batch_size * counter:batch_size * (counter + 1)]
        image_list = []
        mask_list = []
        mask2_list = []
        for file, mask in batch_files:

            image= tiff.imread(file[0])
            
            Landsat = tiff.imread(mask[1])
            Landsat = np.nan_to_num(Landsat)


            image = np.nan_to_num(image)

            image_list.append(image)
            mask2_list.append(Landsat)

        counter += 1
        image_list = np.array(image_list)
        mask2_list = np.array(mask2_list)
        yield (image_list, mask2_list)

        if counter == number_of_batches:
            counter = 0

def mybatch_generator_prediction_bi_unet(tstfiles, img_rows, img_cols, batch_size, max_possible_input_value=1):
    number_of_batches = np.ceil(len(tstfiles) / batch_size)
    counter = 0

    while True:

        beg = batch_size * counter
        end = batch_size * (counter + 1)
        batch_files = tstfiles[beg:end]
        image_list = []

        for file in batch_files:

            image= tiff.imread(file[0])

            image = np.nan_to_num(image)

            image_list.append(image)

        counter += 1
        # print('counter = ', counter)
        image_list = np.array(image_list)

        yield (image_list)

        if counter == number_of_batches:
            counter = 0
