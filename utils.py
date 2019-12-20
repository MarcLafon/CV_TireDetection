# -*- coding: utf-8 -*-
from imgaug.augmentables.bbs import BoundingBox


def targets2txt(bboxe: BoundingBox):
    return '["%s", [%d, %d, %d, %d]]' % (bboxe.label, bboxe.x1, bboxe.y1, bboxe.x2, bboxe.y2)


def save_targets(bboxe: BoundingBox, filepath: str):
    f = open(filepath, "w")
    f.write(targets2txt(bboxe))
    f.close()
