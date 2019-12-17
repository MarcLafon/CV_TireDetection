#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# This software is released under the Apache License 2.0 (the "License"); you may not use the software except in compliance with the License.
# The text of the Apache License 2.0 can be found online at:
# http://www.opensource.org/licenses/apache2.0.php
#

import numpy as np


class ImageClassifier():
    """
    This is a basic example based on random prediction
    """

    def __init__(self):
        print('Classifier : Random')
        pass

    def load(self, path):
        pass

    # image_data argument is a NumPy Array object built with cv2.imread ( image_path ) (BGR channels ordering)
    # returns an array [ Boolean, [Integer,Integer,Integer,Integer], Integer ]
    # where Boolean indicates wether the image contains an wear indicator
    #       [Integer,Integer,Integer,Integer] corresponds to the wear indicator bounding box [xmin,ymin,xmax,yxmax]
    #       Integer holds the wear indicator value ]0..100]
    def predict(self, image_data):

        has_temoin_pred = bool(np.random.randint(0, 2))

        if has_temoin_pred:

            box_temoin_xmin_pred = np.random.randint(0, 2000)
            box_temoin_xmax_pred = np.random.randint(box_temoin_xmin_pred + 1, 2500)
            box_temoin_ymin_pred = np.random.randint(0, 1500)
            box_temoin_ymax_pred = np.random.randint(box_temoin_ymin_pred + 1, 2000)
            usure_temoin_pred = 25 * np.random.randint(0, 5)
        else:
            box_temoin_xmin_pred = box_temoin_xmax_pred = box_temoin_ymin_pred = box_temoin_ymax_pred = 0
            usure_temoin_pred = 0

        return [has_temoin_pred,
                [box_temoin_xmin_pred, box_temoin_xmax_pred, box_temoin_ymin_pred, box_temoin_ymax_pred],
                usure_temoin_pred]
