# -*- coding: utf-8 -*-
import os
from torch.utils.data import Dataset
from PIL import Image
from imgaug.augmentables.bbs import BoundingBox
import pandas as pd
from utils import parse_annotation


class Data(object):
    def __init__(self, img, img_id, fold, label="TemoinNone", bboxe=[0, 0, 0, 0]):
        assert(label in ["TemoinNone", "Temoin25", "Temoin50", "Temoin75", "Temoin100"])
        self.img = img
        self.img_id = img_id
        self.bboxe = BoundingBox(x1=bboxe[0], y1=bboxe[1], x2=bboxe[2], y2=bboxe[3], label=label)
        self.label = self.bboxe.label
        self.fold = fold


class MichelinDataset(Dataset):
    def __init__(self, root, folds, transforms=None):
        """

        :param root: path to folder containing input images and targets
        :param folds: path to csv file containing folds table
        :param transforms:
        """
        self.root = root
        self.folds = folds
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        # Get names of all image and annotation files
        self.files_imgs = list(sorted([img_file for img_file in os.listdir(root) if img_file.endswith(".jpg")]))
        self.files_annotation = list(sorted([img_file for img_file in os.listdir(root) if img_file.endswith(".txt")]))

        # Load cross val folds table
        self.table_folds = pd.read_csv(self.folds, sep=",")

    def __getitem__(self, idx):
        # load images ad masks
        path_img = os.path.join(self.root, self.files_imgs[idx])
        path_annotation = os.path.join(self.root, self.files_annotation[idx])
        img = Image.open(path_img)  # .convert("RGB")
        img_id = self.files_imgs[idx][:-4]
        raw_img_id = "_".join(img_id.split("_")[:2])
        label, bboxe = parse_annotation(path_annotation=path_annotation)
        fold = self.table_folds[self.table_folds["img_id"] == raw_img_id]["fold"].values[0]

        if self.transforms:
            img, bboxe = self.transforms(img, bboxe)

        return Data(img, img_id, fold, label, bboxe)

    def __len__(self):
        return len(self.files_imgs)
