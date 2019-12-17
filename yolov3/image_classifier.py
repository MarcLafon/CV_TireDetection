#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# This code is derived from the ultralytics implementation of Yolov3 (https://github.com/ultralytics/yolov3)
# Credit to Joseph Redmon for YOLO: https://pjreddie.com/darknet/yolo/.
#
# This code sample is released under the Apache License 2.0 (the "License"); you may not use the software except in compliance with the License.
# The text of the Apache License 2.0 can be found online at:
# http://www.opensource.org/licenses/apache2.0.php
#

import numpy as np
import os
import glob
import sys
from PIL import Image
import os.path
from os import path
import time

import cv2
import torch

import models as mods
from utils import *


def _letterbox(img, new_shape=416, color=(127.5, 127.5, 127.5), mode='auto'):
    # Resize a rectangular image to a 32 pixel multiple rectangle
    # https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        ratio = float(new_shape) / max(shape)
    else:
        ratio = max(new_shape) / max(shape)  # ratio  = new / old
    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))

    # Compute padding https://github.com/ultralytics/yolov3/issues/232
    if mode is 'auto':  # minimum rectangle
        dw = np.mod(new_shape - new_unpad[0], 32) / 2  # width padding
        dh = np.mod(new_shape - new_unpad[1], 32) / 2  # height padding
    elif mode is 'square':  # square
        dw = (new_shape - new_unpad[0]) / 2  # width padding
        dh = (new_shape - new_unpad[1]) / 2  # height padding
    elif mode is 'rect':  # square
        dw = (new_shape[1] - new_unpad[0]) / 2  # width padding
        dh = (new_shape[0] - new_unpad[1]) / 2  # height padding

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    top = max(top, 0)
    bottom = max(bottom, 0)
    left = max(left, 0)
    right = max(right, 0)
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)  # resized, no border
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded square
    return img, ratio, dw, dh


class ImageClassifier():
    """
    This is a basic example of prediction based on a neural network
    """

    def __init__(self, img_size=416):
        print('Classifier : Yolo V3')
        self.model = None
        self.img_size = img_size

    def load(self, path):
        device = torch_utils.select_device()
        self.model = mods.Darknet(os.path.join(path, "yolov3_1cls.cfg"))
        # Load weights
        self.model.load_state_dict(
                torch.load(os.path.join(path, "frozen_inference_graph.pt"), map_location=device)['model'])

        # Fuse Conv2d + BatchNorm2d layers
        self.model.fuse()

        # Eval mode
        self.model.to(device).eval()

    # image_data argument is a NumPy Array object built with cv2.imread ( image_path ) (BGR channels ordering)
    # returns an array [ Boolean, [Integer,Integer,Integer,Integer], Integer ]
    # where Boolean indicates wether the image contains an wear indicator
    #       [Integer,Integer,Integer,Integer] corresponds to the wear indicator bounding box [xmin,ymin,xmax,yxmax]
    #       Integer holds the wear indicator value ]0..100]
    def predict(self, image_data):
        with torch.no_grad():
            device = torch_utils.select_device()

            # Padded resize
            img, _, _, _ = _letterbox(image_data, new_shape=self.img_size)

            # Normalize RGB
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
            img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0

            # Get detections
            img = torch.from_numpy(img).unsqueeze(0).to(device)
            pred, _ = self.model(img)

            det = non_max_suppression(pred, .5, .5)[0]

            has_indicator = False
            bbox = [0, 0, 0, 0]
            wears = 0

            if det is not None and len(det) > 0:
                # Rescale boxes from 416 to true image size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image_data.shape).round()

                # sort by confidence
                det = det.cpu().detach().numpy()
                order = np.argsort(det[:, -3])
                det = det[order]

                *xyxy, conf, cls_conf, cls = det[0]
                if conf > .5:
                    has_indicator = True
                    bbox = xyxy
                    wears = 80

            return [has_indicator, bbox, wears]


if __name__ == '__main__':
    imc = ImageClassifier()
    imc.load(path="sample_code_submission_YOLOV3")
    a = imc.predict(image_path="train_dataset\\train_33.jpg")
    print(a)
    a = imc.predict(image_path="20190401_085607.jpg")
    print(a)
