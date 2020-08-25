from __future__ import division, print_function, absolute_import

from timeit import time
from PIL import Image
import warnings
import cv2
import numpy as np
import argparse
import os
import shutil
import imutils.video

from shapely.geometry import Point, Polygon, shape, box
from keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

from libs.deep_sort import preprocessing
from libs.deep_sort import nn_matching
from libs.deep_sort.detection import Detection
from libs.deep_sort.tracker import Tracker
from tools import generate_detections as gdet

from utils import MOI
from utils.parser import get_config
from utils.utils import check_in_polygon, check_number_MOI, init_board, write_board

from src.detect import build_detector_v3
# from src.classify import mobileNet
from videocaptureasync import VideoCaptureAsync


warnings.filterwarnings('ignore')


class VideoTracker(object):
    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args
        self.use_classify = args.use_classify
        self.video_flag = args.video
        self.video_path = args.VIDEO_PATH
        self.track_line = []

        if args.read_detect == 'None':
            self.detector = build_detector_v3(cfg)
            self.classes = self.detector.classes

        if os.path.basename(args.VIDEO_PATH).split('.')[1] == 'txt':
            self.video_name = os.path.basename(args.VIDEO_PATH).split('.')[
                                               0].rstrip("_files")
        else:
            self.video_name = os.path.basename(args.VIDEO_PATH).split('.')[0]
        self.result_filename = os.path.join(
            './logs/output', self.video_name + '_result.txt')

        self.polygon_ROI = Polygon(cfg.CAM.ROI_DEFAULT)
        self.ROI_area = Polygon(shell=self.polygon_ROI).area
        self.TRACKING_ROI = Polygon(cfg.CAM.TRACKING_ROI)
        self.number_MOI = cfg.CAM.NUMBER_MOI

    def run_detection(self, image, encoder, frame_id):
        boxes, confidence, classes = self.detector(image)
        print("[INFO] len boxes: ", len(boxes))
        features = encoder(image, boxes)
        detections = [Detection(bbox, 1.0, cls, feature) for bbox, _, cls, feature in
                      zip(boxes, confidence, classes, features)]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(
            boxes, self.cfg.DEEPSORT.NMS_MAX_OVERLAP, scores)
        detections = [detections[i] for i in indices]
        detections_in_ROI = []

        print("[INFO] detected: ", len(detections))
        for det in detections:
            bbox = det.to_tlbr()
            centroid_det = (int((bbox[0] + bbox[2])//2),
                            int((bbox[1] + bbox[3])//2))
            if check_in_polygon(centroid_det, self.TRACKING_ROI):
                detections_in_ROI.append(det)
        print("[INFO] detections in ROI: ", len(detections_in_ROI))
        logFile = os.path.join(log_detected_cam_dir,
                               'frame_' + str(frame_id) + '.txt')
        with open(logFile, "a+") as f:
            # for det in detections_in_ROI:
            for det in detections:
                bbox = det.to_tlbr()
                score = "%.2f" % round(det.confidence * 100, 2) + "%"

                if len(classes) > 0:
                    cls = det.cls
                    # write log file
                    f.write("{} {} {} {} {} {}\n".format(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]),
                                                         round(det.confidence * 100, 2), cls))

        print("-----------------")
        # return detections_in_ROI
        return detections, detections_in_ROI

    def read_detection(self, image, frame_info, encoder, frame_id):
        detect_folder_path = self.args.read_detect
        detect_file_path = os.path.join(
            detect_folder_path, frame_info + ".txt")

        # file text store path to each frame
        detect_file = open(detect_file_path, 'r')
        lines = detect_file.readlines()

        boxes = []
        confidence = []
        classes = []

        for line in lines:
            detect = line.split()

            bbox = [int(detect[0]), int(detect[1]),
                        int(detect[2]), int(detect[3])]
            score = float(detect[4])
            class_id = int(detect[5])

            boxes.append(bbox)
            confidence.append(score)
            classes.append(class_id)

        features = encoder(image, boxes)
        detections = [Detection(bbox, 1.0, cls, feature) for bbox, _, cls, feature in
                      zip(boxes, confidence, classes, features)]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(
            boxes, self.cfg.DEEPSORT.NMS_MAX_OVERLAP, scores)
        detections = [detections[i] for i in indices]
        detections_in_ROI = []

        print("[INFO] detected: ", len(detections))
        for det in detections:
            bbox = det.to_tlbr()
            centroid_det = (int((bbox[0] + bbox[2])//2),
                            int((bbox[1] + bbox[3])//2))
            if check_in_polygon(centroid_det, self.TRACKING_ROI):
                detections_in_ROI.append(det)
        print("[INFO] detections in ROI: ", len(detections_in_ROI))
        print("-----------------")
        # return detections_in_ROI
        return detections, detections_in_ROI

    def draw_tracking(self, image, tracker, tracking, detections, frame_id, objs_dict):
        if tracking:
            # Call the tracker
            tracker.predict()
            tracker.update(detections)
            print("[INFO] track in ROI: ", len(tracker.tracks))
            print("[INFO] detection in ROI: ", len(detections))

            for det in detections:
                bbox_det = det.to_tlbr()
                cv2.rectangle(image, (int(bbox_det[0]), int(bbox_det[1])), (int(bbox_det[2]), int(bbox_det[3])), (0, 0, 255), 2)
                
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue

                bbox = track.to_tlbr()
                centroid = (int((bbox[0] + bbox[2])//2),
                            int((bbox[1] + bbox[3])//2))

                if track.track_id not in objs_dict:
                    objs_dict.update({track.track_id: {'flag_in_out': 0,
                                                       'best_bbox': track.det_best_bbox,
                                                       'best_bboxconf': track.det_confidence,
                                                       'class_id': track.det_class}})

                # get the first point(x,y) when obj move into ROI
                if len(centroid) !=0 and check_in_polygon(centroid, self.polygon_ROI) and objs_dict[track.track_id]['flag_in_out'] == 0:
                    objs_dict[track.track_id].update({'flag_in_out': 1,
                                                      'point_in': centroid,
                                                      'point_out': None})

                # if bbox conf of obj < bbox conf in new frame ==> update best bbox conf
                if objs_dict[track.track_id]['best_bboxconf'] < track.det_confidence:
                    objs_dict[track.track_id].update({'best_bbox': track.det_best_bbox,
                                                      'best_bboxconf': track.det_confidence,
                                                      'class_id': track.det_class})

                objs_dict[track.track_id]['centroid'] = centroid  # update position of obj each frame
                objs_dict[track.track_id]['last_bbox'] = bbox

                cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
                cv2.putText(image, "ID: " + str(track.track_id), (int(bbox[0]), int(bbox[1])), 0, 1e-3 * image.shape[0], (0, 255, 0), 1)
                cv2.circle(image, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

                print('objs in track list: ', tracker.get_number_obj())
                # draw track line
                image = track.draw_track_line(image)

            # print("[INFO] dict_MOI after added: ", dict_MOI)
            # print("[INFO] dict_detections: after added", dict_detections)

        print("----------------")
        return image, objs_dict

    # def run_classifier(self, clf_model, clf_labels, obj_img):
    #     if len(obj_img) == 0:
    #         return -1
    #     class_id = mobileNet.predict_from_model(obj_img, clf_model, clf_labels)
    #     return int(class_id)

    # def compare_class(self, class_id):
    #     if (class_id >= 0 and class_id <= 4):
    #         class_id = 0
    #     if (class_id > 4 and class_id <= 7):
    #         class_id = 1
    #     if (class_id == 9 or class_id == 10):
    #         class_id = 2
    #     if (class_id == 8 or (class_id <= 13 and class_id > 10)):
    #         class_id = 3
    #     return class_id

    def counting(self, count_frame, cropped_frame, _frame, objs_dict, counted_obj, arr_cnt_class, clf_model=None, clf_labels=None):
        vehicles_detection_list = []
        frame_id = count_frame
        class_id = None
        cv2.putText(_frame, "Frame ID: {}".format(str(frame_id)), (1050, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        for (track_id, info_obj) in objs_dict.items():
            centroid = info_obj['centroid']

            if int(track_id) in counted_obj:  # check if track_id in counted_object ignore it
                continue
             # if track_id not in counted object then check if centroid in range of ROI then count it
            if len(centroid) != 0 and check_in_polygon(centroid, self.polygon_ROI) == False and info_obj['flag_in_out'] == 1:
                info_obj['point_out'] = centroid
                # if self.use_classify:  # clf chua su dung duoc, do cat hinh sai frame!!!!!!!!!!!!!
                #     bbox = info_obj['best_bbox']
                #     obj_img = cropped_frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2]), :] #crop obj following bbox for clf
                #     class_id = self.run_classifier(
                #         clf_model, clf_labels, obj_img)
                #     if class_id == -1:
                #         continue
                # else:
                #     class_id = info_obj['class_id']
                class_id = info_obj['class_id']

                # special class not in contest
                if class_id == 4:
                    continue

                bbox = info_obj['last_bbox']
                obj_img = cropped_frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2]), :]
                image_folder = os.path.join(
                    log_classify_cam_dir, "class_" + str(class_id+1))
                image_file = os.path.join(image_folder, 'frame_' + str(frame_id) + '_' + str(track_id) + '_' + str(class_id) + '.jpg')
                try:
                    cv2.imwrite(image_file, obj_img)
                except:
                    print("Something went wrong at line 260")

                # MOI of obj
                moi  , _ = MOI.compute_MOI(self.cfg, info_obj['point_in'], info_obj['point_out'])

                # draw visual
                cv2.putText(_frame, "Class: {}".format(str(class_id + 1)), (int(bbox[0]), int(bbox[1]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 3)

                counted_obj.append(int(track_id))
                #class_id = self.compare_class(class_id)
                if moi > 0:
                    arr_cnt_class[class_id][moi-1] += 1
                    print("[INFO] arr_cnt_class: \n", arr_cnt_class)
                    vehicles_detection_list.append((frame_id, moi, class_id+1))

        print("--------------")
        return _frame, arr_cnt_class, vehicles_detection_list

    def counting_base_area(self, count_frame, cropped_frame, _frame, objs_dict, counted_obj, arr_cnt_class, clf_model=None, clf_labels=None):
        vehicles_detection_list = []
        frame_id = count_frame
        class_id = None

        for (track_id, info_obj) in objs_dict.items():

            if int(track_id) in counted_obj:  # check if track_id in counted_object ignore it
                continue

            centroid = info_obj['centroid']
            bbox = info_obj['last_bbox']

            obj_poly = box(minx=int(bbox[0]), miny=int(bbox[1]), maxx=int(bbox[2]), maxy=int(bbox[3]))
            obj_area = obj_poly.area

            intersect_area_scale = self.polygon_ROI.intersection(obj_poly).area / obj_area
            print('class id: ', track_id)
            print('intersect area scale: ', intersect_area_scale)

            if intersect_area_scale < self.cfg.CAM.THRESHOLD_AREA and info_obj['flag_in_out'] == 1:
                info_obj['point_out'] = centroid

                # if self.use_classify:  # clf chua su dung duoc, do cat hinh sai frame!!!!!!!!!!!!!
                #     bbox = info_obj['best_bbox']
                #     obj_img = cropped_frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2]), :] #crop obj following bbox for clf
                #     class_id = self.run_classifier(
                #         clf_model, clf_labels, obj_img)
                #     if class_id == -1:
                #         continue
                # else:
                #     class_id = info_obj['class_id']
                class_id = info_obj['class_id']
                # special class not in contest
                if class_id == 4:
                    continue

                bbox = info_obj['last_bbox']
                # draw visual
                cv2.putText(_frame, "Class: {}".format(str(class_id + 1)), (int(bbox[0]), int(bbox[1]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 5)
                

                obj_img = cropped_frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2]), :]
                image_folder = os.path.join(
                    log_classify_cam_dir, "class_" + str(class_id+1))
                image_file = os.path.join(image_folder, 'frame_' + str(frame_id) + '_' + str(track_id) + '_' + str(class_id) + '.jpg')
                try:
                    cv2.imwrite(image_file, obj_img)
                except:
                    print("Something went wrong at line 260")

                # MOI of obj
                moi  , _ = MOI.compute_MOI(self.cfg, info_obj['point_in'], info_obj['point_out'])
                counted_obj.append(int(track_id))

                #class_id = self.compare_class(class_id)
                if moi > 0:
                    arr_cnt_class[class_id][moi-1] += 1
                    print("[INFO] arr_cnt_class: \n", arr_cnt_class)
                    vehicles_detection_list.append((frame_id, moi, class_id+1))

        # ROI_poly = Polygon(shell=[[1, 566], [484, 170], [756, 164], [910, 710]])
        # ROI_area = ROI_poly.area
        # obj_poly = box(minx=712, miny=99, maxx=839, maxy=187)
        # obj_poly2 = box(minx=624, miny=218, maxx=697, maxy=256)
        # obj_poly
        # intersect_area_scale = ROI_poly.intersection(obj_poly2).area / obj_poly2.area
        # print('roi area:', ROI_area)
        # print('obj area:', obj_poly2.area)
        # print('inersec area: ', obj_poly2.intersection(ROI_poly).area)
        # print('intersect area: ', intersect_area_scale)
        # plt.plot(*ROI_poly.exterior.xy)
        # plt.plot(*obj_poly2.exterior.xy)
        # plt.plot(*obj_poly2.intersection(ROI_poly).exterior.xy)
        # plt.show()

        return _frame, arr_cnt_class, vehicles_detection_list


    def write_number(self, image, cnt0=0, cnt1=0, cnt2=0, cnt3=0):
        cv2.putText(image, "Loai_1: {}".format(cnt0), (5, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 1)
        cv2.putText(image, "Loai_2: {}".format(cnt1), (5, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        cv2.putText(image, "Loai_3: {}".format(cnt2), (5, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
        cv2.putText(image, "Loai_4: {}".format(cnt3), (5, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

        return image

    def process(self, frame, count_frame, frame_info, encoder, tracking, tracker, objs_dict, counted_obj, arr_cnt_class, clf_model, clf_labels):
        _frame = np.copy(frame)

        # draw ROI and calibrate lines
        _frame = MOI.config_cam(_frame, self.cfg)

        # draw board
        ROI_board = np.zeros((150, 170, 3), np.int)
        _frame[0:150, 0:170] = ROI_board
        _frame, list_col = init_board(_frame, self.number_MOI)

        # if want to detect in path of original frame
        _frame_height, _frame_width = _frame.shape[:2]
        cropped_frame = np.copy(frame)
        # cv2.rectangle(_frame, (int(frame_width*0), int(_frame_height*0.1)), (int(_frame_width*0.98), int(_frame_height*0.98)), (255, 0, 0), 2)

        print("[INFO] Detecting.....")
        if self.args.read_detect == 'None':
            detections, detections_in_ROI = self.run_detection(
                cropped_frame, encoder, count_frame)
        else:
            print("[INFO] use model")
            detections, detections_in_ROI = self.read_detection(
                cropped_frame, frame_info, encoder, count_frame)

        print("[INFO] Tracking....")
        _, objs_dict = self.draw_tracking(
            _frame, tracker, tracking, detections_in_ROI, count_frame, objs_dict)
        print("[INFO] Counting....")
        if self.args.base_area:
            _frame, arr_cnt_class, vehicles_detection_list = self.counting_base_area(count_frame, cropped_frame, _frame,
                                                                                    objs_dict, counted_obj,
                                                                                    arr_cnt_class, clf_model, clf_labels)
        else:
            _frame, arr_cnt_class, vehicles_detection_list = self.counting(count_frame, cropped_frame, _frame,
                                                                            objs_dict, counted_obj,
                                                                            arr_cnt_class, clf_model, clf_labels)
        # delete counted id
        for track in tracker.tracks:
            if int(track.track_id) in counted_obj:
                track.delete()

        # write result to txt
        with open(self.result_filename, 'a+') as result_file:
            for frame_id, movement_id, vehicle_class_id in vehicles_detection_list:
                result_file.write('{} {} {} {}\n'.format(
                    self.video_name, frame_id, movement_id, vehicle_class_id))

        # write number to scoreboard
        _frame = write_board(_frame, arr_cnt_class, list_col, self.number_MOI)

        return _frame

    def run_video(self):
        # init for classify module
        clf_model = None
        clf_labels = None
        # if self.use_classify:
        #     clf_model, clf_labels = mobileNet.load_model_clf(self.cfg)

        encoder = gdet.create_box_encoder(
            self.cfg.DEEPSORT.MODEL, batch_size=4)
        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", self.cfg.DEEPSORT.MAX_COSINE_DISTANCE, self.cfg.DEEPSORT.NN_BUDGET)
        tracker = Tracker(self.cfg, metric)

        tracking = True
        writeVideo_flag = True
        asyncVideo_flag = False

        list_classes = ['loai_1', 'loai_2', 'loai_3', 'loai_4']
        arr_cnt_class = np.zeros(
            (len(list_classes), self.number_MOI), dtype=int)

        fps = 0.0
        fps_imutils = imutils.video.FPS().start()
        counted_obj = []
        count_frame = 0
        objs_dict = {}

        # file_path = 'data/demo.MOV'
        if asyncVideo_flag:
            video_capture = VideoCaptureAsync(self.video_path)
        else:
            video_capture = cv2.VideoCapture(self.video_path)

        if asyncVideo_flag:
            video_capture.start()

        if writeVideo_flag:
            if asyncVideo_flag:
                w = int(video_capture.cap.get(3))
                h = int(video_capture.cap.get(4))
            else:
                w = int(video_capture.get(3))
                h = int(video_capture.get(4))
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter('output_cam.avi', fourcc, 10, (w, h))
            frame_index = -1

        while True:
            count_frame += 1
            ret, frame = video_capture.read()
            if ret != True:
                break

            frame_info = self.video_name + "_" + str(count_frame - 1)

            t1 = time.time()
            # frame = cv2.flip(frame, -1)

            _frame = self.process(frame, count_frame, frame_info, encoder, tracking, tracker,
                                  objs_dict, counted_obj, arr_cnt_class, clf_model, clf_labels)

            if writeVideo_flag:  # and not asyncVideo_flag:
                # save a frame
                out.write(_frame)
                frame_index = frame_index + 1
            
            # visualize
            if self.args.visualize:
                _frame = imutils.resize(_frame, width=1000)
                cv2.imshow("Final result", _frame)

            fps_imutils.update()

            if not asyncVideo_flag:
                fps = (fps + (1./(time.time()-t1))) / 2
                print("FPS = %f" % (fps))

            # Press Q to stop!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        fps_imutils.stop()
        print('imutils FPS: {}'.format(fps_imutils.fps()))

        if asyncVideo_flag:
            video_capture.stop()
        else:
            video_capture.release()

        if writeVideo_flag:
            out.release()

        cv2.destroyAllWindows()

    def run_img(self):
        # init for classify module
        clf_model = None
        clf_labels = None
        # if self.use_classify:
        #     clf_model, clf_labels = mobileNet.load_model_clf(self.cfg)

        encoder = gdet.create_box_encoder(
            self.cfg.DEEPSORT.MODEL, batch_size=4)
        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", self.cfg.DEEPSORT.MAX_COSINE_DISTANCE, self.cfg.DEEPSORT.NN_BUDGET)
        tracker = Tracker(self.cfg, metric)

        tracking = True
        asyncVideo_flag = False

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output_cam.avi', fourcc, 10, (1280, 720))
        frame_index = -1

        list_classes = ['loai_1', 'loai_2', 'loai_3', 'loai_4']
        arr_cnt_class = np.zeros(
            (len(list_classes), self.number_MOI), dtype=int)

        fps = 0.0
        fps_imutils = imutils.video.FPS().start()
        counted_obj = []
        count_frame = 0
        objs_dict = {}

        # file text store path to each frame
        path_file = open(self.video_path, 'r')
        lines = path_file.readlines()
        txt_name = os.path.basename(self.video_path)
        farther_path = self.video_path.rstrip(txt_name)

        print(farther_path)
        for line in lines:
            line = line.rstrip('\n')
            count_frame += 1
            if len(line) < 5:
                continue
            img_path = os.path.join(farther_path, line)
            print(img_path)
            frame = cv2.imread(img_path)

            frame_info = os.path.basename(img_path).split('.')[0]

            t1 = time.time()
            # frame = cv2.flip(frame, -1)

            _frame = self.process(frame, count_frame, frame_info, encoder, tracking, tracker,
                                  objs_dict, counted_obj, arr_cnt_class, clf_model, clf_labels)

            out.write(_frame)
            frame_index = frame_index + 1

            # visualize
            if self.args.visualize:
                _frame = imutils.resize(_frame, width=1000)
                cv2.imshow("Final result", _frame)

            fps_imutils.update()

            if not asyncVideo_flag:
                fps = (fps + (1./(time.time()-t1))) / 2
                print("FPS = %f" % (fps))

            # Press Q to stop!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        fps_imutils.stop()
        print('imutils FPS: {}'.format(fps_imutils.fps()))
        out.release()
        cv2.destroyAllWindows()


def create_logs_dir():
    if not os.path.exists('logs'):
        os.mkdir('logs')

    log_detected_dir = os.path.join('logs', 'detection')
    if not os.path.exists(log_detected_dir):
        os.mkdir(log_detected_dir)

    log_tracking_dir = os.path.join('logs', 'tracking')
    if not os.path.exists(log_tracking_dir):
        os.mkdir(log_tracking_dir)

    log_output_dir = os.path.join('logs', 'output')
    if not os.path.exists(log_output_dir):
        os.mkdir(log_output_dir)

    log_classify_dir = os.path.join('logs', 'check_classify')
    if not os.path.exists(log_classify_dir):
        os.mkdir(log_classify_dir)

    return log_detected_dir, log_tracking_dir, log_output_dir, log_classify_dir


def create_cam_log(cam_name, log_detected_dir, log_tracking_dir, log_output_dir, log_classify_dir):
    log_detected_cam_dir = os.path.join(log_detected_dir, cam_name)
    if not os.path.exists(log_detected_cam_dir):
        os.mkdir(log_detected_cam_dir)

    log_tracking_cam_dir = os.path.join(log_tracking_dir, cam_name)
    if not os.path.exists(log_tracking_cam_dir):
        os.mkdir(log_tracking_cam_dir)

    log_output_cam_dir = os.path.join(log_output_dir, cam_name)
    if not os.path.exists(log_output_cam_dir):
        os.mkdir(log_output_cam_dir)

    log_classify_cam_dir = os.path.join(log_classify_dir, cam_name)
    if not os.path.exists(log_classify_cam_dir):
        os.mkdir(log_classify_cam_dir)
    for i in range(4):

        folder_clf_class = os.path.join(
            log_classify_cam_dir, "class_" + str(i+1))
        if not os.path.exists(folder_clf_class):
            os.mkdir(folder_clf_class)

    return log_detected_cam_dir, log_tracking_cam_dir, log_output_cam_dir, log_classify_cam_dir


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("VIDEO_PATH", type=str)
    parser.add_argument("--config_detection", type=str,
                        default="./configs/yolov3.yaml")
    parser.add_argument("--config_deepsort", type=str,
                        default="./configs/deep_sort.yaml")
    parser.add_argument("--config_cam", type=str,
                        default="./configs/cam25.yaml")
    parser.add_argument("--use_classify", type=bool, default=False)
    parser.add_argument("--config_classifier", type=str,
                        default="./configs/mobileNet.yaml")
    parser.add_argument("-v", "--visualize", type=bool, default=False)
    parser.add_argument("--video", type=bool, default=False)
    parser.add_argument("--read_detect", type=str, default='None')
    parser.add_argument("--base_area", type=bool, default=False)

    return parser.parse_args()


if __name__ == '__main__':
    if os.path.exists("logs"):
        shutil.rmtree('./logs')

    args = parse_args()
    cfg = get_config()
    # setup code
    cfg.merge_from_file(args.config_detection)
    cfg.merge_from_file(args.config_deepsort)
    cfg.merge_from_file(args.config_cam)
    cfg.merge_from_file(args.config_classifier)

    # create dir/subdir logs
    log_detected_dir, log_tracking_dir, log_output_dir, log_classify_dir = create_logs_dir()

    # create dir cam log
    log_detected_cam_dir, log_tracking_cam_dir, log_output_cam_dir, log_classify_cam_dir = create_cam_log(cfg.CAM.NAME,
                                                                                                          log_detected_dir, log_tracking_dir, log_output_dir, log_classify_dir)


    video_tracker = VideoTracker(cfg, args)
    # video_tracker.counting_base_area()
    print('args.video: ', args.video)
    if args.video:
        print('*****in video-mode*****')
        video_tracker.run_video()
    else:
        print('*****in image-mode*****')
        video_tracker.run_img()
