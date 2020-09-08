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
import math
import matplotlib.pyplot as plt

from collections import deque, Counter 
from shapely.geometry import Point, Polygon, shape, box
from keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder
from sympy import symbols, Eq, solve


from libs.deep_sort import preprocessing
from libs.deep_sort import nn_matching
from libs.deep_sort.detection import Detection
from libs.deep_sort.tracker import Tracker
from tools import generate_detections as gdet

from utils import MOI
from utils.parser import get_config
from utils.utils import check_in_polygon, init_board, write_board, config_cam

from src.detect import build_detector_v3
from videocaptureasync import VideoCaptureAsync


warnings.filterwarnings('ignore')


class VideoTracker(object):
    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args
        self.video_flag = args.video
        self.video_path = args.VIDEO_PATH
        self.track_line = []
        self.count_point = []

        if args.read_detect == 'None':
            self.detector = build_detector_v3(cfg)
            self.classes = self.detector.classes

        if os.path.basename(args.VIDEO_PATH).split('.')[1] == 'txt':
            self.video_name = os.path.basename(args.VIDEO_PATH).split('.')[
                                               0].rstrip("_files")
        else:
            self.video_name = os.path.basename(args.VIDEO_PATH).split('.')[0]

        self.result_filename = os.path.join(
            './data/submission_output', 'submission.txt')

        if args.count == "cosine":
            self.count_method = 1
        elif args.count == "line":
            self.count_method = 2
        elif args.count == "cosine-line":
            self.count_method = 3
        elif args.count == "cosine-line-region":
            self.count_method = 4

        self.polygon_ROI = Polygon(cfg.CAM.ROI_DEFAULT)
        self.ROI_area = Polygon(shell=self.polygon_ROI).area
        # self.TRACKING_ROI = Polygon(cfg.CAM.TRACKING_ROI)
        self.number_MOI = cfg.CAM.NUMBER_MOI

        self.color_list = [(255,0,255), (255,100,0), (0,255,0), (139, 69, 19), (132, 112, 255), (0, 154, 205), (0, 255, 127), 
                            (238, 180, 180), (255, 69, 0), (238, 106, 167), (221, 160, 221), (0, 128, 128)]

    def run_detection(self, image, encoder, frame_id):
        boxes, confidence, classes = self.detector(image)
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
            if check_in_polygon(centroid_det, self.polygon_ROI):
                detections_in_ROI.append(det)

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

        # print("[INFO] detected: ", len(detections))
        for det in detections:
            bbox = det.to_tlbr()
            centroid_det = (int((bbox[0] + bbox[2])//2),
                            int((bbox[1] + bbox[3])//2))
            if check_in_polygon(centroid_det, self.polygon_ROI):
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

            for det in detections:
                bbox_det = det.to_tlbr()
                cv2.rectangle(image, (int(bbox_det[0]), int(bbox_det[1])), (int(bbox_det[2]), int(bbox_det[3])), (0, 0, 255), 1)
                
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
                                                       'class_id': track.det_class,
                                                       'frame': -1,
                                                       'centroid_list': [],
                                                       'class_list': [] }}) # frame when vehicle out ROI BTC (frame estimate)

                # get the first point(x,y) when obj move into ROI
                if len(centroid) !=0 and check_in_polygon(centroid, self.polygon_ROI) and objs_dict[track.track_id]['flag_in_out'] == 0:
                    objs_dict[track.track_id].update({'flag_in_out': 1,
                                                      'point_in': centroid,
                                                      'point_out': None,
                                                      'frame_in': frame_id})

                # if bbox conf of obj < bbox conf in new frame ==> update best bbox conf
                if objs_dict[track.track_id]['best_bboxconf'] < track.det_confidence:
                    objs_dict[track.track_id].update({'best_bbox': track.det_best_bbox,
                                                      'best_bboxconf': track.det_confidence,
                                                      'class_id': track.det_class})

                objs_dict[track.track_id]['centroid'] = centroid  # update position of obj each frame
                objs_dict[track.track_id]['centroid_list'].append(centroid) 
                objs_dict[track.track_id]['last_bbox'] = bbox
                objs_dict[track.track_id]['last_frame'] = frame_id
                objs_dict[track.track_id]['class_list'].append(track.det_class)

                cv2.rectangle(image, (int(bbox[0]), int(bbox[1])-15), (int(bbox[0]+50), int(bbox[1])), (255, 255, 255), -1)
                cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 1)
                cv2.putText(image,str(track.det_class+1) + "." + str(track.track_id), (int(bbox[0]), int(bbox[1])-1), 0, 0.5, (0, 0, 0), 1)
                cv2.circle(image, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

                # draw track line
                image = track.draw_track_line(image)

        print("----------------")
        return image, objs_dict

    # find parameters of line equation
    def line_equation(self, point1, point2):
        a, b = symbols('a b')
        eq1 = Eq(point1[0] * a + b - point1[1])
        eq2 = Eq(point2[0] * a + b - point2[1])

        result = solve((eq1,eq2), (a, b))
        a = result[a]
        c = result[b]
        b = -1
        return a, b, c

    # calculate distance form centroid of obj to ROI line
    def distance_point2roi(self, centroid, point1, point2):
        a, b, c = self.line_equation(point1, point2)
        d = abs((a * centroid[0] + b * centroid[1] + c)) / (math.sqrt(a * a + b * b)) 
        return d

    def estimate_frame(self, point_previous_out, point_out, moi, last_bbox, distance_point_line):
        distance_in_out = math.sqrt((point_out[0] - point_previous_out[0])**2 + (point_out[1] - point_previous_out[1])**2)
        delta_frame = 1          # a.k.a delta time
        velocity = distance_in_out / delta_frame
        acceleration = velocity / delta_frame

        s = distance_point_line
        w = int(last_bbox[2]) - int(last_bbox[0])
        h = int(last_bbox[3]) - int(last_bbox[1])

        if w > h:
            s += w/2
        else:
            s += h/2

        _t = (-velocity) + math.sqrt((velocity ** 2) + (2 * acceleration * s))
        frame_estimate = _t / acceleration
        frame_estimate = round(frame_estimate)
        return frame_estimate

    def voting_class(self, class_list):
        occurence_count = Counter(class_list) 
        return occurence_count.most_common(1)[0][0]

    def find_MOI_candidate(self, region_list, centroid_list):
        MOI_candidate = []
        for index, region in enumerate(region_list):
            region = Polygon(region)
            centroids_in_region = [centroid for centroid in centroid_list if check_in_polygon(centroid, region)]
            percent_point = len(centroids_in_region) / len(centroid_list)
            if percent_point >= self.cfg.CAM.PIM_THRESHOLD:
                MOI_candidate.append(index + 1)
        return MOI_candidate

    def counting(self, count_frame, cropped_frame, _frame, objs_dict, counted_obj, arr_cnt_class, clf_model=None, clf_labels=None):
        vehicles_detection_list = []
        frame_id = count_frame
        class_id = None
        cv2.putText(_frame, "Frame ID: {}".format(str(frame_id)), (1000, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        for (track_id, info_obj) in objs_dict.items():
            centroid = info_obj['centroid']

            if int(track_id) in counted_obj:  # check if track_id in counted_object ignore it
                continue

             # if track_id not in counted object then check if centroid in range of ROI then count it
            if (info_obj['last_frame'] + 2 < count_frame and info_obj['flag_in_out'] == 1) or (check_in_polygon(centroid, self.polygon_ROI) == False and info_obj['flag_in_out'] == 1):
                info_obj['point_out'] = centroid
                class_id = self.voting_class(info_obj['class_list'])

                # ignore special class not in contest
                if class_id == 4:
                    continue

                # compute MOI of obj
                if self.count_method == 1:
                    moi = MOI.compute_MOI_cosine(self.cfg, info_obj['point_in'], info_obj['point_out'])
                elif self.count_method == 2:
                    moi, _, _ = MOI.compute_MOI(self.cfg, info_obj['point_in'], info_obj['point_out'])
                elif self.count_method == 3:
                    moi, _, count = MOI.compute_MOI(self.cfg, info_obj['point_in'], info_obj['point_out'])
                    if count == 0 or count > 1:
                        moi = MOI.compute_MOI_cosine(self.cfg, info_obj['point_in'], info_obj['point_out'])
                elif self.count_method == 4:
                    MOI_candidate = self.find_MOI_candidate(self.cfg.CAM.ROI_SPLIT_REGION, info_obj['centroid_list'])
                    moi, count = MOI.compute_MOI_from_candidate(self.cfg, info_obj['point_in'], info_obj['point_out'], MOI_candidate)
                    if count == 0 or count > 1:
                        moi = MOI.compute_MOI_cosine_from_candidate(self.cfg, info_obj['point_in'], info_obj['point_out'], MOI_candidate)
                
                # mark objs which are counted
                counted_obj.append(int(track_id))

                if moi > 0:
                    info_obj['moi'] = moi
                    track_distance = math.sqrt((info_obj['point_out'][0] - info_obj['point_in'][0]) ** 2 + (info_obj['point_out'][1] - info_obj['point_in'][1]) ** 2)
                    
                    if track_distance < self.cfg.CAM.D_THRESHOLD[moi-1]:
                        continue
                    
                    if self.args.frame_estimate:
                        distance_point_line = self.distance_point2roi(centroid, self.cfg.CAM.LINE_OUT_ROI[moi-1][0], self.cfg.CAM.LINE_OUT_ROI[moi-1][1])
                        info_obj['frame'] = frame_id + self.estimate_frame(info_obj['centroid_deque'][0], info_obj['centroid_deque'][-1], 
                                                                moi, info_obj['last_bbox'], distance_point_line)
                    else:
                        info_obj['frame'] = frame_id

                    # visualize when obj out the ROI
                    if info_obj['frame'] == frame_id:
                        cv2.circle(_frame, (int(centroid[0]), int(centroid[1])), 12, self.color_list[moi-1], -1)
                        cv2.putText(_frame, str(class_id + 1) + '.' + str(track_id), (int(centroid[0]) -3, int(centroid[1])),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
                    arr_cnt_class[class_id][moi-1] += 1
                    vehicles_detection_list.append((info_obj['frame'], moi, class_id+1))

        print("--------------")
        return _frame, arr_cnt_class, vehicles_detection_list

    def counting_base_area(self, count_frame, cropped_frame, _frame, objs_dict, counted_obj, arr_cnt_class, clf_model=None, clf_labels=None):
        vehicles_detection_list = []
        frame_id = count_frame
        class_id = None
        cv2.putText(_frame, "Frame ID: {}".format(str(frame_id)), (1000, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        for (track_id, info_obj) in objs_dict.items():

            if info_obj['frame'] == frame_id:
                class_id = info_obj['class_id']
                # draw visual
                psc = info_obj['point_out']        # point show counting
                cv2.circle(_frame, (int(psc[0]), int(psc[1])), 12, (0, 0, 200), -1)
                cv2.putText(_frame, str(class_id + 1) + '.' + str(track_id), (int(psc[0]) -3, int(psc[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            if int(track_id) in counted_obj:  # check if track_id in counted_object ignore it
                continue

            centroid = info_obj['centroid']
            bbox = info_obj['last_bbox']

            obj_poly = box(minx=int(bbox[0]), miny=int(bbox[1]), maxx=int(bbox[2]), maxy=int(bbox[3]))
            obj_area = obj_poly.area

            intersect_area_scale = self.polygon_ROI.intersection(obj_poly).area / obj_area

            if intersect_area_scale < 0.01 and info_obj['flag_in_out'] == 1:
                info_obj['point_out'] = centroid

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
                moi = MOI.compute_MOI_cosine(self.cfg, info_obj['point_in'], info_obj['point_out'])
                counted_obj.append(int(track_id))

                #class_id = self.compare_class(class_id)
                if moi > 0:
                    info_obj['frame'] = frame_id + self.cfg.CAM.FRAME_MOI[moi-1]
                    arr_cnt_class[class_id][moi-1] += 1
                    vehicles_detection_list.append((frame_id + self.cfg.CAM.FRAME_MOI[moi-1], moi, class_id+1))

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
        _frame = config_cam(_frame, self.cfg, self.args)

        # draw board
        ROI_board = np.zeros((150, 170, 3), np.int)
        _frame[0:150, 0:170] = ROI_board
        _frame, list_col = init_board(_frame, self.number_MOI)

        # if want to detect in path of original frame
        _frame_height, _frame_width = _frame.shape[:2]
        cropped_frame = np.copy(frame)

        print("[INFO] Detecting.....")
        if self.args.read_detect == 'None':
            detections, detections_in_ROI = self.run_detection(
                cropped_frame, encoder, count_frame)
        else:
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

        encoder = gdet.create_box_encoder(
            self.cfg.DEEPSORT.MODEL, batch_size=4)
        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", self.cfg.DEEPSORT.MAX_COSINE_DISTANCE, self.cfg.DEEPSORT.NN_BUDGET)
        tracker = Tracker(self.cfg, metric)

        tracking = True
        writeVideo_flag = self.args.out_video
        asyncVideo_flag = False

        list_classes = ['loai_1', 'loai_2', 'loai_3', 'loai_4']
        arr_cnt_class = np.zeros(
            (len(list_classes), self.number_MOI), dtype=int)

        fps = 0.0
        fps_imutils = imutils.video.FPS().start()
        counted_obj = []
        count_frame = 0
        objs_dict = {}

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
            output_camname = 'output' + '_' + self.video_name + '.avi'
            out = cv2.VideoWriter(output_camname, fourcc, 10, (1280, 720))
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

        encoder = gdet.create_box_encoder(
            self.cfg.DEEPSORT.MODEL, batch_size=4)
        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", self.cfg.DEEPSORT.MAX_COSINE_DISTANCE, self.cfg.DEEPSORT.NN_BUDGET)
        tracker = Tracker(self.cfg, metric)

        tracking = True
        asyncVideo_flag = False

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_camname = 'output' + '_' + self.video_name + '.avi'
        out = cv2.VideoWriter(output_camname, fourcc, 10, (1280, 720))
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
    if not os.path.exists('data'):
        os.mkdir('data')

    log_output_dir = os.path.join('data', 'submission_output')
    if not os.path.exists(log_output_dir):
        os.mkdir(log_output_dir)
    return log_output_dir

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("VIDEO_PATH", type=str)
    parser.add_argument("--config_detection", type=str,
                        default="./configs/yolov3.yaml")
    parser.add_argument("--config_deepsort", type=str,
                        default="./configs/deep_sort.yaml")
    parser.add_argument("--config_cam", type=str,
                        default="./configs/cam6.yaml")
    parser.add_argument("-v", "--visualize", type=bool, default=False)
    parser.add_argument("--out_video", type=bool, default=False)
    parser.add_argument("--video", type=bool, default=True)
    parser.add_argument("--read_detect", type=str, default="None")
    parser.add_argument("--base_area", type=bool, default=False)
    parser.add_argument("-f", "--frame_estimate", type=bool, default=False)
    parser.add_argument("-c", "--count", type=str, default="cosine-line-region")

    return parser.parse_args()


if __name__ == '__main__':
    if os.path.exists("data/submission_output"):
        shutil.rmtree('./data/submission_output')

    args = parse_args()
    cfg = get_config()
    # setup code
    cfg.merge_from_file(args.config_detection)
    cfg.merge_from_file(args.config_deepsort)
    cfg.merge_from_file(args.config_cam)

    # create dir/subdir logs
    log_output_dir = create_logs_dir()

    video_tracker = VideoTracker(cfg, args)
    # video_tracker.counting_base_area()
    print('args.video: ', args.video)
    if args.video:
        print('*****in video-mode*****')
        video_tracker.run_video()
    else:
        print('*****in image-mode*****')
        video_tracker.run_img()
