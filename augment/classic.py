# -*- coding: utf-8 -*-
import os
import tqdm
import shutil
import pathlib
from utils import save_targets
from torchvision import transforms
from torchvision.transforms import ToTensor, ToPILImage
from load import MichelinDataset, Data
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import numpy as np
from torchvision.transforms import functional as F


class CustomRandomErasing(transforms.RandomErasing):

    def __init__(self, name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name

    def __call__(self, image: np.array, bounding_boxes: BoundingBox):
        """
        Args:
            image (np.array):
            bounding_boxes (BoundingBox):

        Returns:
            img (np.array): Erased np.array image.
        """
        torch_image = ToTensor()(image)
        if np.random.uniform(0, 1) < self.p:
            a, b = self.ratio
            ratio = [(a, b), (1/b, 1/a)][np.random.choice([0,1])]
            occlusion = 1.
            while occlusion > 0.2:
                x, y, h, w, v = self.get_params(torch_image, scale=self.scale, ratio=ratio, value=self.value)
                bboxe_erased = BoundingBox(x1=y, y1=x, x2=y + w, y2=x + h, label="boxe_erased")
                if bounding_boxes.intersection(bboxe_erased):
                    occlusion = bounding_boxes.intersection(bboxe_erased).area / bounding_boxes.area
                else:
                    occlusion = 0.
            img = ToPILImage()(F.erase(torch_image, x, y, h, w, v, self.inplace))
            return np.array(img), bounding_boxes
        return image, bounding_boxes


def main_classic(path_input, path_output):
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
    dataset = MichelinDataset(path_input)
    flipud = iaa.Flipud(p=1, name="_flipud")
    fliplr = iaa.Fliplr(p=1, name="_fliplr")
    rotate = iaa.Affine(rotate=[-25, -15, -10, 10, 15, 25], mode="edge", name="_rotate")
    erase = CustomRandomErasing(p=1, scale=(0.1, 0.2), ratio=(2, 6), value="random", name="_erase")
    shear = iaa.Affine(shear=[-25, -15, 15, 25], mode="edge", name="_shear")
    convert2gray = iaa.Grayscale(alpha=0.8, name="_gray")
    rand_sat_hue = iaa.AddToHueAndSaturation(value_hue=(-20, 20), value_saturation=(-20, 20),
                                             per_channel=.8, name="_sat_hue")
    brigthen = iaa.Multiply((0.5, 1.5), name="_bright")
    contrast = iaa.CLAHE(clip_limit=(2, 4), name="_contrast")
    noise = iaa.SaltAndPepper(0.1, per_channel=False, name="_noise")
    operations = [fliplr, flipud, rotate, erase, shear, convert2gray, rand_sat_hue, brigthen, contrast, noise]
    resize = iaa.Resize({"height": 320, "width": 416})
    for data in dataset:
        # raw_img, bboxe = resize(image=np.array(data.img), bounding_boxes=data.bboxe)
        raw_img, bboxe = np.array(data.img), data.bboxe
        ToPILImage()(raw_img).save(os.path.join(path_output, "%s.jpg" % data.img_id))
        save_targets(bboxe, os.path.join(path_output, "%s.txt" % data.img_id))
        for i, operation in enumerate(operations):
            if np.random.rand() < 1:
                aug_img, aug_bboxe = operation(image=raw_img, bounding_boxes=bboxe)
                aug_img = ToPILImage()(aug_img)
                aug_img.save(os.path.join(path_output, "%s%s.jpg" % (data.img_id, operation.name)))
                save_targets(aug_bboxe, os.path.join(path_output, "%s%s.txt" % (data.img_id, operation.name)))


if __name__ == "__main__":
    try:
        project_home = str(pathlib.Path(__file__).resolve().parent.parent)
    except NameError:
        project_home = globals()['_dh'][0]

    path_input_data = os.path.join(project_home, "data", "data_fivecrop")
    if "data_fivecrop" not in os.listdir(os.path.join(project_home, "data")):
        raise FileNotFoundError("the folder data_fivecrop doese not exists, launch the script fivecrop.py")

    path_data_augment = os.path.join(project_home, "data", "data_classic")
    if "data_classic" not in os.listdir(os.path.join(project_home, "data")):
        os.mkdir(path_data_augment)

    main_classic(path_input_data, path_data_augment)
