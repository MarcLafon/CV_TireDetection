# -*- coding: utf-8 -*-
import argparse
import sys
import os
import shutil
import pathlib
from utils import save_targets
from torchvision import transforms
from load import MichelinDataset, Data
# import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox


def bbx_out_img_zeroing(x1, y1, x2, y2, img_shape):
    w, h = img_shape
    # x2 <0 --> bboxe en dehors à gauche de l'image
    # y2 <0 --> bboxe en dehors en haut de l'image
    # x1 >w --> bboxe en dehors à droite de l'image
    # y1 >h --> bboxe en dehors en bas de l'image
    if (x2 <= 0) or (y2 <= 0) or (x1 >= w) or (y1 >= h):
        return 0, 0, 0, 0
    else:
        if x2 >= w:
            x2 = w - 1
        if y2 >= h:
            y2 = h - 1
        if x1 <= 0:
            x1 = 0
        if y1 <= 0:
            y1 = 0
        return x1, y1, x2, y2


def five_crop_bbx(position, bboxe, input_shape, output_shape):
    valid_positions = ["top_left", "top_right", "bottom_left", "bottom_rigtht", "center"]
    if isinstance(position, int):
        position = valid_positions[position]
    assert (position in ["top_left", "top_right", "bottom_left", "bottom_rigtht", "center"])
    x1, y1, x2, y2 = bboxe.x1, bboxe.y1, bboxe.x2, bboxe.y2
    w, h = input_shape
    _w, _h = output_shape
    dw, dh = (w - _w), (h - _h)
    if position == "center":
        x1 -= dw / 2
        y1 -= dh / 2
        x2 -= dw / 2
        y2 -= dh / 2
    elif position == "top_right":
        x1 -= dw
        x2 -= dw
    elif position == "bottom_left":
        y1 -= dh
        y2 -= dh
    elif position == "bottom_rigtht":
        x1 -= dw
        y1 -= dh
        x2 -= dw
        y2 -= dh

    x1, y1, x2, y2 = bbx_out_img_zeroing(x1, y1, x2, y2, output_shape)
    _bboxe = BoundingBox(x1=int(x1), x2=int(x2), y1=int(y1), y2=int(y2))
    if _bboxe.is_out_of_image(output_shape):
        _bboxe.label = "TemoinNone"
    else:
        _bboxe.label = bboxe.label
    return _bboxe


def fivecrop(data_: Data, path: str, factor=1.5):
    w, h = data_.img.size
    _w, _h = (int(w / factor), int(h / factor))

    tpl_img, tpr_img, bl_img, br_img, center_img = transforms.FiveCrop((_h, _w))(data_.img)

    # Top left image
    tpl_bbx = five_crop_bbx(0, data_.bboxe, (w, h), (_w, _h))
    tpl_img.save(os.path.join(path, "%s_tpl.jpg" % data_.img_id))
    save_targets(tpl_bbx, os.path.join(path, "%s_tpl.txt" % data_.img_id))

    # Top right image
    tpr_bbx = five_crop_bbx(1, data_.bboxe, (w, h), (_w, _h))
    tpr_img.save(os.path.join(path, "%s_tpr.jpg" % data_.img_id))
    save_targets(tpr_bbx, os.path.join(path, "%s_tpr.txt" % data_.img_id))

    # Bottom left image
    bl_bbx = five_crop_bbx(2, data_.bboxe, (w, h), (_w, _h))
    bl_img.save(os.path.join(path, "%s_bl.jpg" % data_.img_id))
    save_targets(bl_bbx, os.path.join(path, "%s_bl.txt" % data_.img_id))

    # Bottom right image
    br_bbx = five_crop_bbx(3, data_.bboxe, (w, h), (_w, _h))
    br_img.save(os.path.join(path, "%s_br.jpg" % data_.img_id))
    save_targets(br_bbx, os.path.join(path, "%s_br.txt" % data_.img_id))

    # Center image
    center_bbx = five_crop_bbx(4, data_.bboxe, (w, h), (_w, _h))
    center_img.save(os.path.join(path, "%s_center.jpg" % data_.img_id))
    save_targets(center_bbx, os.path.join(path, "%s_center.txt" % data_.img_id))


if __name__ == "__main__":
    # ** Setting Up Parser ** #
    parser = argparse.ArgumentParser(description='Script performing data augmentation on Michelin dataset')
    parser.add_argument('--init', help='', action='store_true')
    parser.add_argument('--path_raw_data', default="training_dataset")
    # ** Parsing arguments ** #
    #sys.argv = [""]  # for debugging purpose, delete this line when the script is finished
    args = parser.parse_args()
    init = args.init

    # ** Code Starts Here **

    try:
        project_home = str(pathlib.Path(__file__).resolve().parent)
    except NameError:
        project_home = globals()['_dh'][0]

    path_raw_data = os.path.join(project_home, "training_dataset")
    path_data_augment = os.path.join(project_home, "augmented_data")

    print(project_home)
    print(path_raw_data)
    print(path_data_augment)

    dataset = MichelinDataset(path_raw_data)

    if pathlib.Path(path_data_augment).parts[-1] not in os.listdir(project_home):
        os.mkdir(path_data_augment)

    if init:
        for data in dataset:
            shutil.copy(os.path.join(path_raw_data, data.img_id + ".jpg"),
                        os.path.join(path_data_augment, data.img_id + ".jpg"))
            shutil.copy(os.path.join(path_raw_data, data.img_id + ".txt"),
                        os.path.join(path_data_augment, data.img_id + ".txt"))
            fivecrop(data, path_data_augment, factor=1.5)
