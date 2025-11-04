import keras
import keras.backend as K
from tqdm import tqdm
import os

class ADAMLearningRateTracker(keras.callbacks.Callback):
    """It prints out the last used learning rate after each epoch (useful for resuming a training)
    original code: https://github.com/keras-team/keras/issues/7874#issuecomment-329347949
    """

    def __init__(self, end_lr):
        super(ADAMLearningRateTracker, self).__init__()
        self.end_lr = end_lr

    def on_epoch_end(self, epoch, logs={}):  # works only when decay in optimizer is zero
        optimizer = self.model.optimizer
        # t = K.cast(optimizer.iterations, K.floatx()) + 1
        # lr_t = K.eval(optimizer.lr * (K.sqrt(1. - K.pow(optimizer.beta_2, t)) /
        #                               (1. - K.pow(optimizer.beta_1, t))))
        # print('\n***The last Actual Learning rate in this epoch is:', lr_t,'***\n')
        print('\n***The last Basic Learning rate in this epoch is:', K.eval(optimizer.lr), '***\n')
        # stops the training if the basic lr is less than or equal to end_learning_rate
        if K.eval(optimizer.lr) <= self.end_lr:
            print("training is finished")
            self.model.stop_training = True



def get_input_image_names_bi(directory_name, if_train=True):
    list_img = []
    list_msk = []
    list_test_ids = []

    if if_train:

        for each_image in os.listdir(directory_name + '/CloudyLandsat8'):
            fl_img = []
            
            fl_img1 = directory_name + '/CloudyLandsat8/' + each_image
            fl_img2 = directory_name + '/CloudyLandsat8/' + each_image
            fl_img3 = directory_name + '/CloudyLandsat8/' + each_image

            fl_img.append(fl_img1)
            fl_img.append(fl_img2)
            fl_img.append(fl_img3)

            list_img.append(fl_img)
            
            fl_msk = []
            fl_msk1 = directory_name + '/Mask/' + each_image
            fl_msk2 = directory_name + '/T/' + each_image
            fl_msk.append(fl_msk1)
            fl_msk.append(fl_msk2)
            list_msk.append(fl_msk)
    else:

        for each_image in os.listdir(directory_name + '/CloudyLandsat8'):
            fl_id = each_image
            list_test_ids.append(fl_id)
            fl_img = []
            
            fl_img1 = directory_name + '/CloudyLandsat8/' + each_image
            fl_img2 = directory_name + '/CloudyLandsat8/' + each_image
            fl_img3 = directory_name + '/CloudyLandsat8/' + each_image

            fl_img.append(fl_img1)
            fl_img.append(fl_img2)
            fl_img.append(fl_img3)

            list_img.append(fl_img)


    if if_train:
        return list_img , list_msk
    else:
        return list_img, list_test_ids
