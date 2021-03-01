import os
import urllib.request
import zipfile
from tqdm import tqdm

from segtrain.eval_model import evaluate_segmentation_network, batch_predict
from segtrain.train_model import train_segmentation_network
from segtrain.trainer.Config import Config


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def make_file(file_path):
    directory = os.path.dirname(file_path)

    if not os.path.exists(directory):
        os.makedirs(directory)

def makeddir(path):
    if not os.path.exists(path):
        os.makedirs(path)



def download_zipfile(zip_file_url, out_dir):
    local_file_path = os.path.join(out_dir, os.path.basename(zip_file_url))
    download_url(zip_file_url, local_file_path)
    with zipfile.ZipFile(os.path.join(out_dir, os.path.basename(zip_file_url)), 'r') as zip_images:
        zip_images.extractall(out_dir)


class ISICConfig(Config):
    """Configuration for training on the demo mnist segmentation  dataset.
    Derives from the base Config class and overrides some values.
    """

    # Give the configuration a recognizable name unique to an experiment.
    NAME = "unet_isic"

    # the directory where the output of the experiments will be saved.
    LOG_ROOT_DIR = 'temp'

    # size of input image as (HEIGHT, WIDTH) format. To enable dynamic size use (None, None)
    IMAGE_SIZE = (256, 256)
    # extension of the input image
    IMAGE_EXT = ['.jpg']
    _LABEL_MULTIPLIER = 1/255.0

    # Flag denoting if the input image should be loaded in BGR format. If False, the image will be loaded as graysale
    IS_RGB = True

    # Number of classes
    NUM_CLASSES = 1 + 1  # 10 digits + background

    # Number of training steps per epoch, when None, it will be calculated from data size
    STEPS_PER_EPOCH = None

    LEARNING_RATE = 0.0001
    LEARNING_MOMENTUM = 0.9

    # Number of training epochs
    NUM_EPOCH = 100

    # Training mini-batch size. If IMAGE_SIZE is set to (None, None) then BATCH_SIZE should be 1
    BATCH_SIZE = 16

    # Validation batch size
    VAL_BATCH_SIZE = 4

    # A function that takes a file name, and returns a subject id or a group id. This is used while splittig data into
    # train/val/test such that same group falls on a same split. Set the value to staticmethod(None), if no grouping is needed

    image_grouping_function = [None]  # [lambda x: x.split('_')[0]]

    # directory where training images are saved.
    IMG_DIR = ['path_tp_images']

    # directory where training label images are saved.
    # The label file should be in .png format and have same name as images in IMG_DIR folder.
    LABEL_DIR = ['path_to_labels']

    TRAIN_VAL_TEST_RATIOS = [(0.2, 0.3, 0.3)]
    # Flag denoting if the data split files are to be used. Following split files will be searched
    # in IMG_DIR, 'train.txt', 'val.txt', 'test.txt'.
    USE_DATA_SPLIT_FILES = False

    DROPOUT_RATE = 0.2
    VERBOSE = 1

    ##Augmentation parameters
    AUG_FLIP_HORZ = True

    AUG_FLIP_VERT = True

    # Rotation angle range in degrees x: wil be sampled from -x to x
    AUG_ROTATE = 5
    # When random resize option is enabled, batch size should be 1
    RANDOM_RESIZE = False


import glob
import shutil


def fix_gtimages_names(gt_dir, new_gt_dir):
    makeddir(new_gt_dir)
    files = glob.glob(gt_dir + '/*.png')
    for f in files:
        newname = '_'.join(os.path.basename(f).split('_')[:2]) + '.png'
        shutil.copy(f, os.path.join(new_gt_dir, newname))


if __name__ == '__main__':
    ##Download the data
    train_zip_url = 'http://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Training_Data.zip'
    gt_zip_url = 'http://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Training_GroundTruth.zip'
    data_dir = 'data/'  # '/Users/ssedai/data/isic/'
    makeddir(data_dir)
    print('Downloading  ', gt_zip_url)

    download_zipfile(train_zip_url, data_dir)
    download_zipfile(gt_zip_url, data_dir)
    fix_gtimages_names(os.path.join(data_dir, 'ISBI2016_ISIC_Part1_Training_GroundTruth'),
                       os.path.join(data_dir, 'ISBI2016_ISIC_Part1_Training_GroundTruthNew'))
    config = ISICConfig()
    config.IMG_DIR = [os.path.join(data_dir, 'ISBI2016_ISIC_Part1_Training_Data')]
    config.LABEL_DIR = [os.path.join(data_dir, 'ISBI2016_ISIC_Part1_Training_GroundTruthNew')]
    train_segmentation_network(config)
    evaluate_segmentation_network(config)

    # predict on any images in a given model
    # batch_predict('mnist_data/temp', 'mnist_data/temp_out', config=config,image_extension='.jpeg')
