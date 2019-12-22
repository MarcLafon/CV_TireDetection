# -*- coding: utf-8 -*-
import argparse
import os
import pathlib
from load import MichelinDataset
from augment.fivecrop import main_fivecrop
import imgaug.augmenters as iaa
import numpy as np

if __name__ == "__main__":
    # ** Setting Up Parser ** #
    parser = argparse.ArgumentParser(description='Script performing data augmentation on Michelin dataset')
    parser.add_argument('--init', help='', action='store_true')
    parser.add_argument('--classic', help='', action='store_true')
    parser.add_argument('--cut', help='', action='store_true')
    parser.add_argument('--augmix', help='', action='store_true')
    parser.add_argument('--style', help='', action='store_true')
    parser.add_argument('--path_raw_data', default="training_dataset")
    # ** Parsing arguments ** #
    # sys.argv = ["--init","--classic"]  # for debugging purpose, delete this line when the script is finished
    args = parser.parse_args()
    init = args.init
    classic = args.classic
    cut = args.cut
    augmix = args.augmix
    style = args.style

    # ** Code Starts Here **
    try:
        project_home = str(pathlib.Path(__file__).resolve().parent.parent)
    except NameError:
        project_home = globals()['_dh'][0]

    path_raw_data = os.path.join(project_home, "training_dataset")
    path_data_augment = os.path.join(project_home, "data_fivecrop")
    if "data_fivecrop" not in os.listdir(project_home):
        os.mkdir(path_data_augment)

    if init:
        main_fivecrop(path_raw_data, path_data_augment)

    if classic:
        # Performing classical data augmentation :
        # Techniques classiques :
        # - horizontal ﬂipping -> _flipud_
        # - verical flipping -> _fliplr_
        # - random small rotations -> _rotate_
        # - random erasing (https://github.com/zhunzhong07/Random-Erasing)-> _erase_
        # - random shearing -> _shear_
        # - conversion to grayscale (?) -> _gray_
        # - random perturbations of hue -> _hue_
        # - random perturbations of saturation -> sat_
        # - random perturbations of brightness ->_light_
        # - random perturbations of contrast ->_contr_
        # - random noise -> _noise_
        dataset = MichelinDataset(path_data_augment)
        rotation = iaa.Affine(rotate=[-35, -25, -15, 15, 25, 35], mode="edge")
        img_aug, bbx_aug = rotation(image=np.array(dataset[10].img), bounding_boxes=dataset[10].bboxe)
        img_bbx = BoundingBoxesOnImage([bbx_aug],shape=img_aug.shape)
        plt.imshow(img_bbx.draw_on_image(img_aug,size=6))
        plt.imshow(np.array(dataset[10].img))