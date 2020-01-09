# -*- coding: utf-8 -*-
from imgaug.augmentables.bbs import BoundingBox
import numpy as np
import pandas as pd


def targets2txt(bboxe: BoundingBox):
    return '["%s", [%d, %d, %d, %d]]' % (bboxe.label, bboxe.x1, bboxe.y1, bboxe.x2, bboxe.y2)


def save_targets(bboxe: BoundingBox, filepath: str):
    f = open(filepath, "w")
    f.write(targets2txt(bboxe))
    f.close()


def parse_annotation(path_annotation):
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


def extract_bboxe_img(img, bboxe: BoundingBox):
    raw_img, bboxe = np.array(img), bboxe
    x1, y1, w, h = bboxe.x1, bboxe.y1, bboxe.width, bboxe.height
    return raw_img[y1:y1 + h, x1:x1 + w, :]
