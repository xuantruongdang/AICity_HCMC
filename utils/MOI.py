import cv2
import numpy as np
import imutils
import argparse
import os 

class Point: 
    def __init__(self, x, y): 
        self.x = x 
        self.y = y 

def config_cam(img, cfg):
    arr = cfg.CAM.ROI_DEFAULT
    arr1 = cfg.CAM.TRACKING_ROI

    pts = np.array(arr)
    pts1 = np.array(arr1)

    # plot ROI
    cv2.drawContours(img, [pts], -1, (0, 0, 255), 2)
    cv2.drawContours(img, [pts1], -1, (0, 255, 0), 2)

    # plot calibrations lines
    line1 = cfg.CAM.LINE1
    line2 = cfg.CAM.LINE2

    line1_startX = line1[0][0]
    line1_startY = line1[0][1]
    line1_endX = line1[1][0]
    line1_endY = line1[1][1]

    line2_startX = line2[0][0]
    line2_startY = line2[0][1]
    line2_endX = line2[1][0]
    line2_endY = line2[1][1]

    cv2.line(img, (line1_startX, line1_startY), (line1_endX, line1_endY),(0, 255, 0))
    cv2.line(img, (line2_startX, line2_startY), (line2_endX, line2_endY),(150, 0, 0))

    return img

# check p, q, r for alignment or not
def onSegment(p, q, r): 
    if ( (q.x <= max(p.x, r.x)) and (q.x >= min(p.x, r.x)) and 
           (q.y <= max(p.y, r.y)) and (q.y >= min(p.y, r.y))): 
        return True
    return False
  
# find orientation 
def orientation(p, q, r):        
    val = (float(q.y - p.y) * (r.x - q.x)) - (float(q.x - p.x) * (r.y - q.y)) 
    if (val > 0): 
        return 1    # counter-clockwise

    elif (val < 0): 
        return 2    # clockwise 

    else: 
        return 0    # alignment
  
# check p1q1 intersect p2q2 or not
def doIntersect(p1,q1,p2,q2):    

    o1 = orientation(p1, q1, p2) 
    o2 = orientation(p1, q1, q2) 
    o3 = orientation(p2, q2, p1)    
    o4 = orientation(p2, q2, q1) 
  
    if ((o1 != o2) and (o3 != o4)): 
        return True
  
    if ((o1 == 0) and onSegment(p1, p2, q1)): 
        return True
  
    if ((o2 == 0) and onSegment(p1, q2, q1)): 
        return True
  
    if ((o3 == 0) and onSegment(p2, p1, q2)): 
        return True
  
    if ((o4 == 0) and onSegment(p2, q1, q2)): 
        return True
  
    # If none of the cases 
    return False

def transform(point_1, point_2):
    point_1 = Point(point_1[0], point_1[1])
    point_2 = Point(point_2[0], point_2[1])

    return point_1, point_2

def compute_MOI(cfg, point_in, point_out):
    line1 = cfg.CAM.LINE1
    line2 = cfg.CAM.LINE2

    line1_leftpoint = line1[0]
    line1_rightpoint = line1[1]
    line2_leftpoint = line2[0]
    line2_rightpoint = line2[1]

    # transform to Point format
    p2, q2 = transform(point_in, point_out)
    line1_leftpoint, line1_rightpoint = transform(line1_leftpoint, line1_rightpoint)
    line2_leftpoint, line2_rightpoint = transform(line2_leftpoint, line2_rightpoint)

    moi = -1
    orient = None

    # Vector V has first point p2 and last point q2: present the movement of vehicle
    # if V intersect line1, MOI = 1
    if doIntersect(line1_leftpoint, line1_rightpoint, p2, q2):
        moi = 1
        orient = orientation(line1_leftpoint, line1_rightpoint, p2)

    # if V intersect line2, MOI = 2
    if doIntersect(line2_leftpoint, line2_rightpoint, p2, q2):
        moi = 2
        orient = orientation(line2_leftpoint, line2_rightpoint, p2)

    return moi, orient