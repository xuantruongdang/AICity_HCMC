import cv2
import numpy as np
import os
import random
import imutils
import time 
import argparse

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
from collections import deque

from utils import MOI
from utils.parser import get_config
from utils.utils import check_in_polygon, check_number_MOI, init_board, write_board

from shapely.geometry import Point, Polygon

net = cv2.dnn.readNet("models/yolov3.weights", "configs/yolov3.cfg")
output_path = os.path.join("results", "my_result.jpg")
# Name custom object
classes =  ['di_bo','xe_dap','xe_may','xe_hang_rong','xe_ba_gac','xe_taxi','xe_hoi','xe_ban_tai','xe_cuu_thuong','xe_khach','xe_buyt','xe_tai','xe_container','xe_cuu_hoa']

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize a list of colors to represent each possible class label
np.random.seed(100)
COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")
 
def detect_yolo(img):
    height, width, channels = img.shape

    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    return boxes, class_ids

def process(video_path, cfg):
    # start detect video
    pts = [deque(maxlen=30) for _ in range(9999)]

    max_cosine_distance = 0.7
    nn_budget = 100
    nms_max_overlap = 0.3

    counter = []
    model_filename = 'models/market1501.pb'
    encoder = gdet.create_box_encoder(model_filename ,batch_size=4)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)


    cap = cv2.VideoCapture(video_path)
    while (True):
        ret, frame = cap.read()
        height, width, channels = frame.shape

        _frame = MOI.config_cam(frame, cfg)


        # try:
        boxes, class_names = detect_yolo(_frame)
        features = encoder(_frame, boxes)
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxes, features)]
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)
        print("[INFO] track in ROI: ", len(tracker.tracks))
            
        i = int(0)
        indexIDs = []
        c = []
        boxes = []
        for det in detections:
            bbox = det.to_tlbr()
            cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            #boxes.append([track[0], track[1], track[2], track[3]])
            # indexIDs.append(int(track.track_id))
            # counter.append(int(track.track_id))
            bbox = track.to_tlbr()

            #bbox_center_point(x,y)
            center = (int(((bbox[0])+(bbox[2]))/2),int(((bbox[1])+(bbox[3]))/2))

            # check track in ROI
            # if not check_in_polygon(center, ROI_POLYGON):
            #     continue
            
            indexIDs.append(int(track.track_id))
            counter.append(int(track.track_id))

            # create color
            color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]

            # draw track
            cv2.rectangle(_frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(color), 2)
            cv2.putText(_frame,str(track.track_id),(int(bbox[0]), int(bbox[1] -50)),0, 5e-3 * 150, (color),2)
            if len(class_names) > 0:
               class_name = class_names[0]
               cv2.putText(_frame, str(class_names[0]),(int(bbox[0]), int(bbox[1] -20)),0, 5e-3 * 150, (color),2)
            
            # increase count
            i += 1

            #track_id[center]
            pts[track.track_id].append(center)
            thickness = 2
            #center point
            cv2.circle(_frame,  (center), 1, color, thickness)

	        #draw motion path
            for j in range(1, len(pts[track.track_id])):
                if pts[track.track_id][j - 1] is None or pts[track.track_id][j] is None:
                   continue
                thickness = int(np.sqrt(64 / float(j + 1)) * 2)
                cv2.line(_frame,(pts[track.track_id][j-1]), (pts[track.track_id][j]),(color),thickness)
                #cv2.putText(frame, str(class_names[j]),(int(bbox[0]), int(bbox[1] -20)),0, 5e-3 * 150, (255,255,255),2)

        # except:
        #     pass
        
        cv2.imshow('detection', _frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default="data/cam_18.mp4")
    # parser.add_argument("--config_detection", type=str, default="./configs/yolov3.yaml")
    # parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    parser.add_argument("--config_cam", type=str, default="./configs/cam18.yaml")
    # parser.add_argument("--use_classify", type=bool, default=False)
    # parser.add_argument("--config_classifier", type=str, default="./configs/mobileNet.yaml")
    # parser.add_argument("-v", "--visualize", type=bool, default=False)

    return parser.parse_args()


if __name__ == '__main__':
    if not os.path.exists("results"):
        os.mkdir("results")

    args = parse_args()
    cfg = get_config()

    cfg.merge_from_file(args.config_cam)

    ROI_POLYGON = Polygon(cfg.CAM.ROI_DEFAULT)

    process(args.video_path, cfg)