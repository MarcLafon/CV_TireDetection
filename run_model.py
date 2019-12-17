#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# This software is released under the Apache License 2.0 (the "License"); you may not use the software except in compliance with the License.
# The text of the Apache License 2.0 can be found online at:
# http://www.opensource.org/licenses/apache2.0.php
#

import numpy as np
import pickle
import os
import sys
import glob
import cv2
import yaml


#
# INGESTION
#

def load_solution(reftxt_path):
    refdata = [False, [0, 0, 0, 0], 0]
    try:
        if not os.path.exists(reftxt_path):
            # print ( 'no txt file named ' + reftxt_path )
            return refdata
        f = open(reftxt_path, 'r')
        ref = eval(f.readlines()[0])  # read content as an array
        f.close()
        txtvalue = ref[0][6:]  # get value skipping prefix 'Temoin'
        if txtvalue != 'None':
            xmin = ref[1][0]
            ymin = ref[1][1]
            xmax = ref[1][2]
            ymax = ref[1][3]
            refdata = [True, [xmin, ymin, xmax, ymax], int(txtvalue)]
        # print ( 'values loaded from ' + reftxt_path )
        # print ( str(refdata) )
        # print ( '\n' )
    except:
        pass
    return refdata


def ingestion():
    print(" --- Begin ingestion MICHELIN ---")

    # External class provided by participant
    sys.path.append(submission_dir)
    from image_classifier import ImageClassifier

    ###########################################################################
    # DEBUG
    DEBUG = False
    if DEBUG:
        # Print /tmp/codalab content
        epath = '/tmp/codalab/'
        files = []
        # r=root, d=directories, f = files
        for r, d, f in os.walk(epath):
            for file in f:
                print(os.path.join(r, file))
    ###########################################################################

    # If you want to test the code with your images, just add images files to the PATH_TO_TEST_IMAGES_DIR.
    PATH_TO_TEST_IMAGES_DIR = input_dir

    TEST_IMAGE_PATHS = glob.glob(os.path.join(PATH_TO_TEST_IMAGES_DIR, "*.jpg"))
    # TEST_SOLUTION_PATHS = glob.glob(os.path.join(input_dir, "*.txt"))
    assert len(TEST_IMAGE_PATHS) > 0, 'No image found in `{}`.'.format(PATH_TO_TEST_IMAGES_DIR)
    # assert len(TEST_SOLUTION_PATHS) > 0, 'No txt found in `{}`.'.format(input_dir)
    # print("path test image: ",TEST_IMAGE_PATHS)
    # print("path test solution: ", TEST_SOLUTION_PATHS)

    # num_classes = 5

    classifier = ImageClassifier()

    classifier.load(submission_dir)

    data = []

    for i, image_path in enumerate(TEST_IMAGE_PATHS):
        image_data_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)  # returns image with channels stored in BGR order
        prediction_data = classifier.predict(image_data_bgr)
        solution_data = load_solution(os.path.join(input_dir, os.path.basename(image_path)[:-3] + "txt"))
        data.append([prediction_data, solution_data])

    # print(data)

    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir)

    filename = 'data'
    data_file = open(os.path.join(prediction_dir, filename), 'wb')

    with data_file as fp:
        pickle.dump(data, fp)

    data_file.close()

    print(" --- End ingestion MICHELIN ---")


#
# SCORING
# 

def get_weights(has_temoin, temoin_value):
    # Scoring is weighted to give each class the same value.
    # Warning : distribution of class may be different on the competition dataset.
    if has_temoin:
        return 1 / 250
    return 1 / 125


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def evaluate_prediction(has_temoin_pred, has_temoin_true,
                        box_temoin_xmin_pred, box_temoin_xmin_true,
                        box_temoin_xmax_pred, box_temoin_xmax_true,
                        box_temoin_ymin_pred, box_temoin_ymin_true,
                        box_temoin_ymax_pred, box_temoin_ymax_true,
                        usure_temoin_pred, usure_temoin_true):
    """
    Returns the evaluation of the candidate submission.

    INPUT
    has_temoin_pred : boolean. Prediction of the candidate of the presence of a temoin.
    has_temoin_true : boolean. The true value.

    box_temoin_xmin_pred : int. Prediction of the first side of the box, its minimum X in the referential of
                    the original image. Similar for xmax, ymin, ymax
    box_temoin_xmin_true : int. The corresponding true value. Can be left to zero if there is no temoin

    usure_temoin_pred : float. PRedicted level of usure. must be between 0 (completely worn off) and 1 (new).
    usure_temoin_pred : float. True level of usure. Can be left to zero if there is no temoin

    RETURN :
    a score between 0 and 1

    """

    if not has_temoin_true:
        score = int(not has_temoin_pred)
        return score

    iou = bb_intersection_over_union(
            [box_temoin_xmin_pred, box_temoin_xmax_pred, box_temoin_ymin_pred, box_temoin_ymax_pred],
            [box_temoin_xmin_true, box_temoin_xmax_true, box_temoin_ymin_true, box_temoin_ymax_true])
    # print ( 'Scoring : intersection box ratio %f' % iou )
    """if usure_temoin_true < 25:
        iou_min_threshold = 0.1
    elif usure_temoin_true < 50:
        iou_min_threshold = 0.25
    else :
        iou_min_threshold = 0.5"""

    iou_min_threshold = 0.2

    if iou < iou_min_threshold:
        score = 0
        return score

    if usure_temoin_pred == usure_temoin_true:
        score = 1
        return score

    if np.abs(usure_temoin_pred - usure_temoin_true) < 25:
        score = 0.5
        return score

    return 0


def scoring():
    print(' --- Begin scoring MICHELIN ---')

    #### INPUT/OUTPUT: Get input and output directory names

    if not os.path.exists(score_dir):
        os.makedirs(score_dir)

    # print ( 'Creating score_dir' )

    score_file = open(os.path.join(score_dir, 'scores.txt'), 'w')
    data_file = open(os.path.join(prediction_dir, 'data'), 'rb')

    # print ( 'Loading predicted and solution data files' )

    with data_file as fp:
        data = pickle.load(fp)

    # print ( 'Computing score' )

    total_score = 0
    for i in range(len(data)):
        score = evaluate_prediction(data[i][0][0], data[i][1][0],
                                    data[i][0][1][0], data[i][1][1][0],
                                    data[i][0][1][1], data[i][1][1][1],
                                    data[i][0][1][2], data[i][1][1][2],
                                    data[i][0][1][3], data[i][1][1][3],
                                    data[i][0][2], data[i][1][2])
        # print(score)
        weight = get_weights(has_temoin=data[i][1][0], temoin_value=data[i][1][2])
        total_score += score * weight

    # print(data)
    print('Size of scoring data : %d' % len(data))
    print('Score : %f' % total_score)

    score_file.write("prediction_score: {:0.12f}\n".format(float(total_score)))

    try:
        metadata = yaml.load(open(os.path.join(prediction_dir, 'metadata'), 'r'))
        score_file.write("Duration: {:0.6f}\n".format(metadata['elapsedTime']))
    except:
        score_file.write("Duration: 0\n")
    score_file.close()
    data_file.close()

    print(' --- End scoring MICHELIN ---')


# MAIN

root_dir = "./"

input_dir = root_dir + "training_dataset"
submission_dir = root_dir + "model"
prediction_dir = root_dir + "results"
score_dir = root_dir + "scoring"

if __name__ == "__main__":
    if len(sys.argv) == 1:  # Use the default input and output directories if no arguments are provided
        print('usage: %s <model directory>' % sys.argv[0])
        sys.exit(1)
    else:
        submission_dir = os.path.join(root_dir, sys.argv[1])
        ingestion()
        scoring()
