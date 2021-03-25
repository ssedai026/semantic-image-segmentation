import copy
import glob
import os

import cv2
import numpy as np
import scipy.io as sio
from PIL import Image


## This script convert oct layer annotation to pixelwise labels to train pixelwise semantic segmentation


def mkdir(path, *args):
    """
    Gives a root path and and subpath, makes directory all the way from root to subpaths if they do not exist
    :param path: root path
    :param args:
    :return:
    """
    if(not os.path.exists(path)):
        os.mkdir((path))
    new_path = path
    for dir in args:
        new_path = os.path.join(new_path, dir)
        # print (new_path)
        if (not os.path.exists(new_path)):
            os.mkdir(new_path)
    return new_path



def annonate2(image, layer1, layer2):
    """
    Converts layer points to mask
    :param image:
    :param layer1:
    :param layer2:
    :return:
    """
    mask_filled = convert2mask(image, [layer1], [layer2])
    mask_filled = np.asarray(mask_filled, dtype=np.uint8)
    return image, mask_filled


def convert2mask(image, layer1_pts, layer2_pts):


    contour = []
    contour.extend(layer1_pts)
    layer2_pts_cp = copy.deepcopy(layer2_pts)
    layer2_pts_cp[0].reverse()
    contour.extend(layer2_pts_cp)

    mask_new = np.zeros(np.shape(image))

    # (N,2)
    contour_arr = np.vstack(contour)
    # (N,1,2)
    contour_arr = np.expand_dims(contour_arr, 1)
    cnt_list = []
    cnt_list.append(contour_arr)

    # print 'cnt_list', np.shape(cnt_list)

    contour_arr = contour_arr.astype(np.int32)
    # print contour_arr

    cv2.drawContours(mask_new, [contour_arr], -1, 255, -1)  # 3=-1
    return mask_new


def layers2mask(image, layers):
    masks_all = []
    for i in range(len(layers) - 1):
        _, maski = annonate2(image, layers[i], layers[i + 1])
        masks_all.append(maski)

    _, maskbg_inv = annonate2(image, layers[0], layers[len(layers) - 1])
    mask_bg = 255 - maskbg_inv
    masks_all.append(mask_bg)
    # labelmap= np.argmax(mask_all, axis = 2)
    return masks_all


class OCTVolReader:
    def __init__(self, path):
        self.path = path
        self.file_names = []
        self.load()

    def load(self):
        print (self.path + '/*.mat')
        self.file_names = glob.glob(self.path + '/*.mat')

    def get_data(self):
        for f in self.file_names:
            data = sio.loadmat(f)
            yield [data.get('images'), data.get('layerMaps'), os.path.basename(f)]


def get_pts(lm):
    layers = []
    x = np.asarray(list(range(lm.shape[0])))
    for i in range(lm.shape[1]):
        bdr = lm[:, i]
        bdr_f = bdr[~np.isnan(bdr)]
        x_f = x[~np.isnan(bdr)]
        pts = zip(x_f, bdr_f)
        layers.append(pts)
    return layers


class CropImageLayers:
    def __init__(self, ds):
        self.ds = ds

    def get_data(self):
        for im, layers, fname in self.ds.get_data():

            layersc = []
            for layer in layers:
                layer = list(layer)
                xleft = layer[0][0]
                print(xleft)
                xright = layer[len(layer) - 1][0]
                layerc = [(x - xleft, y) for x, y in layer]
                layersc.append(layerc)
            imc = im[:, xleft:xright]

            yield imc, layersc, fname, [xleft, xright]


class Layers2Mask:
    def __init__(self, ds):
        self.ds = ds

    def get_data(self):
        for im, layers, fname, crop_params in self.ds.get_data():
            masks = layers2mask(im, layers)
            masks = np.expand_dims(masks, -1)
            masks = np.concatenate(masks, axis=2)
            labelmap = np.argmax(masks, axis=2)
            yield im, labelmap, fname,crop_params


class Unstack:
    def __init__(self, ds, sel_frames=None):
        self.ds = ds
        self.sel_frames = sel_frames

    def get_data(self):
        for imstack, lstack, fname in self.ds.get_data():

            if (self.sel_frames is not None): assert type(self.sel_frames) is list
            indices = range(imstack.shape[2]) if self.sel_frames is None else self.sel_frames
            for i in indices:
                im = imstack[:, :, i]
                # 1000x3
                lm = lstack[i, :, :]
                lmf = lm[~np.isnan(lm[:, 0])]
                if (len(lmf) > 0):
                    layers = get_pts(lm)
                    sp_fname = fname.split('.')
                    newfname = sp_fname[0] + '_' + str(i) + '.' + sp_fname[1]

                    #added 26 sept list
                    layers = [list(l) for l in layers]
                    if (len(layers[0]) > 500):
                        yield im, layers, newfname


def showlm(im, lm):
    lm = lm.astype(np.int32)
    Image.fromarray(lm * 50).show()
    Image.fromarray(im).show()


def save_labels(im, lm, fname, img_out_path, label_out_path, crop_params_save_dir, crop_params):
    base_name = fname.split('.')[0]
    cv2.imwrite(os.path.join(img_out_path, base_name+'.jpeg'), im)
    cv2.imwrite(os.path.join(label_out_path, base_name + '.png'), lm)
    np.savetxt(os.path.join(crop_params_save_dir, base_name + '.txt'), np.asanyarray(crop_params))





def prepare_oct_data(raw_datapath, labels_save_path):
    """
    Prepares the training and validation data from duke oct images
    :param raw_datapath:  the location where raw oct files are saved
    #the .mat files cane be downloaded from
    # https://www.dropbox.com/sh/23gzbj7jxhruesa/AABDKD9O9zuTgNsGpG25nn62a
    # https://www.dropbox.com/sh/t0jz33utxlfi6op/AACdpRaUroDQllW9b8dCBKdFa
    :param labels_save_path:  location to save the processed images and labels
    :return:
    """



    image_save_dir = mkdir(labels_save_path,'images')
    labels_save_dir =mkdir(labels_save_path, 'labels')
    crop_params_save_dir = mkdir(labels_save_path, 'crop_params')

    oct = OCTVolReader(raw_datapath)
    oct = Unstack(oct, sel_frames=[40, 45, 50, 55, 60, 35,70, 20])
    oct = CropImageLayers(oct)
    oct = Layers2Mask(oct)
    cnt = 0
    for im, lm, fname, crop_params in oct.get_data():
        print (im.shape, lm.shape, cnt)  # ,len(lm[0]), len(lm[1]), len(lm[2])
        save_labels(im,lm,fname, image_save_dir, labels_save_dir,  crop_params_save_dir, crop_params)
        cnt = cnt + 1




if (__name__ == '__main__'):
    ## This script convert oct layer annotation to pixelwise labels to train pixelwise semantic segmentation

    # First download the data from the following locations:
    # https://www.dropbox.com/sh/23gzbj7jxhruesa/AABDKD9O9zuTgNsGpG25nn62a
    # https://www.dropbox.com/sh/t0jz33utxlfi6op/AACdpRaUroDQllW9b8dCBKdFa

    #Save the downloaded  .mat files in raw_datapath
    raw_datapath = 'data/dukeoct_raw/'
    labels_save_path = 'data/dukeoct_processed'

    prepare_oct_data(raw_datapath, labels_save_path)