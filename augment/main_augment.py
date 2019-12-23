# -*- coding: utf-8 -*-
import argparse
import os
import pathlib
from load import MichelinDataset
from augment.fivecrop import main_fivecrop
from augment.classic import main_classic
import imgaug.augmenters as iaa
import numpy as np

if __name__ == "__main__":
    # ** Setting Up Parser ** #
    parser = argparse.ArgumentParser(description='Script performing data augmentation on Michelin dataset')
    parser.add_argument('--init', help='', action='store_true')
    parser.add_argument('--classic', help='', action='store_true')
    parser.add_argument('--cutmix', help='', action='store_true')
    parser.add_argument('--augmix', help='', action='store_true')
    parser.add_argument('--style', help='', action='store_true')
    parser.add_argument('--path_raw_data', default="training_dataset")
    # ** Parsing arguments ** #
    # sys.argv = ["--init","--classic"]  # for debugging purpose, delete this line when the script is finished
    args = parser.parse_args()
    init = args.init
    classic = args.classic
    cutmix = args.cutmix
    augmix = args.augmix
    style = args.style

    # ** Code Starts Here **
    try:
        project_home = str(pathlib.Path(__file__).resolve().parent.parent)
    except NameError:
        project_home = globals()['_dh'][0]

    path_raw_data = os.path.join(project_home, "data", "training_dataset")
    path_data_five_crop = os.path.join(project_home, "data", "data_fivecrop")
    if "data_fivecrop" not in os.listdir(os.path.join(project_home, "data")):
        os.mkdir(path_data_five_crop)
    path_data_classic = os.path.join(project_home, "data", "data_classic")
    if "data_classic" not in os.listdir(os.path.join(project_home, "data")):
        os.mkdir(path_data_classic)

    if init:
        main_fivecrop(path_raw_data, path_data_five_crop)

    if classic:
        main_classic(path_data_five_crop, path_data_classic)