import argparse
import os
import numpy as np
import cv2

from src.detect import build_detector_v3

from utils.parser import get_config
from utils.utils import check_in_polygon

from libs.deep_sort.detection_yolo import Detection_YOLO

def run_detection(cfg, detector, image, area, log_detected_cam_dir):
    boxes, confidence, classes = detector(image)
    detections_in_ROI = []

    detections = [Detection_YOLO(bbox, confidence, cls) for bbox, confidence, cls in
                      zip(boxes, confidence, classes)]

    # to tlbr
    boxes = np.array([d.to_tlbr() for d in detections])
    classes = np.array([compare_class(int(d.cls)) for d in detections])

    # print("[INFO] detected: ", len(detections))
    for index, bbox in enumerate(boxes):
        centroid_det = (int((bbox[0] + bbox[2])//2), int((bbox[1] + bbox[3])//2))
        if check_in_polygon(centroid_det, cfg.polygon_ROI):
            area[classes[index]].append(compute_area(int(bbox[2] - bbox[0]), int(bbox[1] - bbox[3])))
            
    print("[INFO] detections in ROI: ", len(detections_in_ROI))
    logFile = os.path.join(log_detected_cam_dir, 'classes_area.txt')
    with open(logFile, "a+") as f:
        for class_id, values in enumerate(area):
            values = np.array(values)
            min_value = np.min(values)
            max_value = np.max(values)
            mean_value = np.mean(values)            
                # write log file
            f.write("{} {} {} {}\n".format(class_id, min_value, max_value, mean_value))
            

def compare_class(class_id):
    if (class_id >= 0 and class_id <= 4):
        class_id = 0
    if (class_id > 4 and class_id <= 7):
        class_id = 1
    if (class_id == 9 or class_id == 10):
        class_id = 2
    if (class_id == 8 or (class_id <= 13 and class_id > 10)):
        class_id = 3
    return class_id

def create_logs_dir():
    if not os.path.exists('logs'):
        os.mkdir('logs')

    log_detected_dir = os.path.join('logs', 'area')
    if not os.path.exists(log_detected_dir):
        os.mkdir(log_detected_dir)

    return log_detected_dir

def create_cam_log(cam_name, log_detected_dir):
    log_detected_cam_dir = os.path.join(log_detected_dir, cam_name)
    if not os.path.exists(log_detected_cam_dir):
        os.mkdir(log_detected_cam_dir)
    return log_detected_cam_dir

def compute_area(width, height):
    return width * height

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("VIDEO_PATH", type=str)
    parser.add_argument("--config_detection", type=str, default="./configs/yolov3.yaml")
    parser.add_argument("--config_cam", type=str, default="./configs/cam1.yaml")
    parser.add_argument("--use_classify", type=bool, default=False)
    parser.add_argument("--config_classifier", type=str, default="./configs/mobileNet.yaml")

    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()
    cfg = get_config()
    # setup code
    cfg.merge_from_file(args.config_detection)
    cfg.merge_from_file(args.config_cam)
    cfg.merge_from_file(args.config_classifier)

    detector = build_detector_v3(cfg)

    log_detected_dir = create_logs_dir()
    log_detected_cam_dir = create_cam_log(cfg.CAM.NAME, log_detected_dir)
    
    img_path = ''
    image = cv2.imread(img_path)
    run_detection(cfg, detector, image, area,log_detected_cam_dir)