# -*- coding: utf-8 -*-
import os
from torch.utils.data import Dataset
from PIL import Image
from imgaug.augmentables.bbs import BoundingBox


class Data(object):
    def __init__(self, img, img_id, label="TemoinNone", bboxe=[0, 0, 0, 0]):
        assert(label in ["TemoinNone", "Temoin25", "Temoin50", "Temoin75", "Temoin100"])
        self.img = img
        self.img_id = img_id
        self.bboxe = BoundingBox(x1=bboxe[0], y1=bboxe[1], x2=bboxe[2], y2=bboxe[3], label=label)
        self.label = self.bboxe.label


class MichelinDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned

        self.files_imgs = list(sorted([img_file for img_file in os.listdir(root) if img_file.endswith(".jpg")]))
        self.files_annotation = list(sorted([img_file for img_file in os.listdir(root) if img_file.endswith(".txt")]))

    def __getitem__(self, idx):
        # load images ad masks
        path_img = os.path.join(self.root, self.files_imgs[idx])
        path_annotation = os.path.join(self.root, self.files_annotation[idx])
        img = Image.open(path_img)  # .convert("RGB")
        label, bboxe = self.parse_annotation(path_annotation=path_annotation)

        if self.transforms:
            img, bboxe = self.transforms(img, bboxe)

        return Data(img,
                    self.files_imgs[idx][:-4],
                    label,
                    bboxe)

    def __len__(self):
        return len(self.files_imgs)

    def parse_annotation(self, path_annotation):
        f = open(path_annotation, 'r')
        content = f.read()
        infos = content.split(',')

        label = infos[0]
        label = label.replace('"', "").replace('[', "")

        xmin = int(infos[1].replace("[", "").lstrip())
        ymin = int(infos[2].lstrip())
        xmax = int(infos[3].lstrip())
        ymax = int(infos[4].replace("]", "").lstrip())

        f.close()
        return label, [xmin, ymin, xmax, ymax]