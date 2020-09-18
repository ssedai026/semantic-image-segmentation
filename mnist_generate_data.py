import csv
import os

import cv2
import numpy as np
from tensorflow.keras.datasets import mnist

from segtrain.trainer.visializeoutput_checkpoint import stack_patches



def generate_data(images, labels):
    labels = labels[:, np.newaxis, np.newaxis]

    masks = (images > 64).astype(np.uint8)
    zz = masks * (labels + 1)
    zz = zz - 1
    # replace -1 to 10

    idx = np.where(zz == 255)
    zz[idx] = 10
    zz = np.expand_dims(zz, -1)

    images = np.expand_dims(images, -1)
    pot_im = stack_patches(images, 4, 4)
    pot_label = stack_patches(zz, 4, 4)

    pot_im = np.squeeze(pot_im)
    pot_label = np.squeeze(pot_label)
    #viz_labelmaps = visualize_labelmaps(pot_label, N_class=11).astype(np.uint8)
    return pot_im, pot_label


def write_array2csv(arr, filename):
    with open(filename, 'w') as f:
        lines = map(lambda x: x + '\n', arr)
        f.writelines(lines)





if (__name__ == "__main__"):

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    save_dir = 'mnist_data'
    if (not os.path.exists(save_dir)):
        os.mkdir(save_dir)

    image_dir = os.path.join(save_dir, 'images')
    label_dir = os.path.join(save_dir, 'labels')
    if (not os.path.exists(label_dir)):
        os.mkdir(label_dir)

    if (not os.path.exists(image_dir)):
        os.mkdir(image_dir)
    train_list = []
    for i in range(20000):
        idx = np.random.randint(x_train.shape[0] - 17)
        images = x_train[idx: idx + 16]
        labels = y_train[idx: idx + 16]

        im, label = generate_data(images, labels)

        cv2.imwrite(os.path.join(image_dir, 'train_' + str(i) + '.jpeg'), im)
        cv2.imwrite(os.path.join(label_dir, 'train_' + str(i) + '.png'), label)
        train_list.append('train_' + str(i) + '.jpeg')
    val_list = []
    for i in range(0, 1000):
        idx = np.random.randint(x_test.shape[0] - 17)
        images = x_test[idx: idx + 16]
        labels = y_test[idx: idx + 16]

        im, label = generate_data(images, labels)
        cv2.imwrite(os.path.join(image_dir, 'val_' + str(i) + '.jpeg'), im)
        cv2.imwrite(os.path.join(label_dir, 'val_' + str(i) + '.png'), label)
        val_list.append('val_' + str(i) + '.jpeg')

    test_list = []
    for i in range(0, 1000):
        idx = np.random.randint(x_test.shape[0] - 17)
        images = x_test[idx: idx + 16]
        labels = y_test[idx: idx + 16]

        im, label = generate_data(images, labels)
        cv2.imwrite(os.path.join(image_dir, 'test_' + str(i) + '.jpeg'), im)
        cv2.imwrite(os.path.join(label_dir, 'test_' + str(i) + '.png'), label)
        test_list.append('test_' + str(i) + '.jpeg')

    #parent = os.path.abspath(os.path.join(image_dir, os.pardir))
    write_array2csv(train_list, os.path.join(image_dir, 'train.txt'))
    write_array2csv(val_list, os.path.join(image_dir, 'val.txt'))
    write_array2csv(test_list, os.path.join(image_dir, 'test.txt'))




    print('Segmentation training and validation images are saved in:', image_dir, ' and labels are saved in:',
          label_dir)
    print('Done')
