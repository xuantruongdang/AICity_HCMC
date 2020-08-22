import os
import cv2
import json
import random
import itertools
import numpy as np
import argparse
import cv2

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg as config_detectron

path_weigth = "models/model_fasterRCNN_Khang_10k_46_v3.pth"
path_config = "detectron2/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
confidences_threshold = 0.5
num_of_class = 5

classes = ['Loai1', 'Loai2', 'Loai3', 'Loai4', 'Loai5']

detectron = config_detectron()
detectron.MODEL.DEVICE= 'cpu'
detectron.merge_from_file(path_config)
detectron.MODEL.WEIGHTS = path_weigth

detectron.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidences_threshold
detectron.MODEL.ROI_HEADS.NUM_CLASSES = num_of_class

predictor = DefaultPredictor(detectron)

def predict(image, predictor, file_path):
    outputs = predictor(image)

    boxes = outputs['instances'].pred_boxes
    scores = outputs['instances'].scores
    classes = outputs['instances'].pred_classes

    list_boxes = []
    list_scores = []
    list_classes = []

    for i in range(len(classes)):
        if (scores[i] > 0.5):
            for j in boxes[i]:
                x = int(j[0])
                y = int(j[1])
                w = int(j[2]) - x
                h = int(j[3]) - y

            score = float(scores[i])
            class_id = int(classes[i])
            # list_boxes.append([x, y, w, h])
            # list_scores.append(score)
            # list_classes.append(class_id)

            # cv2.rectangle(image, (x, y), (x+w, y+h), (random.randint(
            #     0, 255), random.randint(0, 255), 255), 1)
            with open(file_path, "a+") as f:
                f.write("{} {} {} {} {} {}\n".format(x, y, w, h, score, class_id))

    return file_path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file_txt", type=str, required=True)
    parser.add_argument("-o", "--output_dir", type=str, default="det_dir")

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    file_path = args.file_txt
    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    with open(file_path) as f:
        for line in f:
            frame = line.rstrip("\n")
            image = cv2.imread(frame)
            frame_name = (frame.split("/")[-1]).split(".")[0]
            frame_path = os.path.join(output_dir, frame_name + '.txt')
            
            file_path = predict(image, predictor, frame_path)

            print("[INFO] Done ", file_path)

    # _frame = cv2.imread("frame.jpg")
    # outputs = predict(_frame, predictor)

    # cv2.imshow("frame", outputs)
    # cv2.waitKey(0)
