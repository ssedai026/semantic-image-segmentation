

from segtrain.eval_model import evaluate_segmentation_network, batch_predict
from segtrain.train_model import train_segmentation_network
from segtrain.trainer.Config import Config



class RPEDCSeg(Config):
    """Configuration for training on the OCT layers (RNFL layer, RPE Drusen complex and Brusch mebrane) from Duke dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name unique to an experiment.
    NAME = "unet_rpedc"

    #the directory where the output of the experiments will be saved.
    LOG_ROOT_DIR='out_octseg'

    # size of input image as (HEIGHT, WIDTH) format. To enable dynamic size use (None, None)
    IMAGE_SIZE = (512,512)
    # extension of the input image
    IMAGE_EXT=['.jpeg']

    #Flag denoting if the input image should be loaded in BGR format. If False, the image will be loaded as graysale
    IS_RGB= False

    #Number of classes
    NUM_CLASSES = 2+1 #2 classes + background

    # Number of training steps per epoch, when None, it will be calculated from data size
    STEPS_PER_EPOCH = 1000

    LEARNING_RATE = 0.0001
    LEARNING_MOMENTUM = 0.9

    #Number of training epochs
    NUM_EPOCH= 200

    #Training mini-batch size. If IMAGE_SIZE is set to (None, None) then BATCH_SIZE should be 1
    BATCH_SIZE=1

    #Validation batch size
    VAL_BATCH_SIZE = 4

    # A function that takes a file name, and returns a subject id or a group id. This is used while splittig data into
    #train/val/test such that same group falls on a same split. Set the value to staticmethod(None), if no grouping is needed

    image_grouping_function = [ lambda x: x.split('_')[5] ]

    #directory where training images are saved. Initialize it using constructer
    IMG_DIR   = ['dukeoct_processed/images']

    # directory where training label images are saved. Initialize it using constructer
    # The label file should be in .png format and have same name as images in IMG_DIR folder.
    LABEL_DIR = ['dukeoct_processed/labels']


    TRAIN_VAL_TEST_RATIOS= [(0.8,0.1, 0.1)]
    DROPOUT_RATE=0.2
    VERBOSE=1

    ##Augmentation parameters
    #Left right flipping
    AUG_FLIP_HORZ = True

    #top down flipping
    AUG_FLIP_VERT = False

    #Rotation angle range in degrees x: wil be sampled from -x to x
    AUG_ROTATE = 6
    #When random resize option is enabled, batch size should be 1
    RANDOM_RESIZE = False




if __name__ == '__main__':

    #The dataset needs to be prepared before running this script.
    #Run oct_layer_prepare_data.py to generate the training/validation set. Note that the OCT .mat files needs to be downloaded
    # in raw_datapath before running the data preperation script

    config = RPEDCSeg()
    model = train_segmentation_network(config)
    #the training logs and models are saved in LOG_ROOT_DIR='out_octseg'
    evaluate_segmentation_network(config)

