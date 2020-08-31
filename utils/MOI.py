import cv2
import numpy as np
import imutils
import argparse
import os 
import random 
from sklearn.metrics.pairwise import cosine_similarity

class Point: 
    def __init__(self, x, y): 
        self.x = x 
        self.y = y 

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
    line = cfg.CAM.LINE
    line_leftpoint = []
    line_rightpoint = []

    for i in line:
      line_leftpoint.append (i[0])
      line_rightpoint.append (i[1])

    # transform to Point format
    p2, q2 = transform(point_in, point_out)
    for i in range(len(line_leftpoint)):
      line_leftpoint[i], line_rightpoint[i] = transform (line_leftpoint[i], line_rightpoint[i])

    moi = -1
    orient = None

    # Vector V has first point p2 and last point q2: present the movement of vehicle
    # if V intersect line1, MOI = 1
    for i in range (len (line_leftpoint)):
      if (doIntersect(line_leftpoint[i], line_rightpoint[i], p2, q2)):
        moi = i + 1
        orient = orientation (line_leftpoint[i], line_rightpoint[i], p2)

    return moi, orient

def compute_cosine(cfg, point_in, point_out):
    MOI = cfg.CAM.MOI
    cosine_results = []

    # vector create by vehicle
    vector_obj = np.array([point_out[0] - point_in[0], point_out[1] - point_in[1]])
    vector_obj = vector_obj.reshape(1, 2)

    for moi in MOI:
        vector_moi = np.array([moi[1][0] - moi[0][0], moi[1][1] - moi[0][1]])
        vector_moi = vector_moi.reshape(1,2)
        cosine = cosine_similarity(vector_moi, vector_obj)
        cosine_results.append(cosine)
    
    return cosine_results

def compute_MOI_cosine(cfg, point_in, point_out):
    cosine_results = compute_cosine(cfg, point_in, point_out)
    min_cosine = np.amax(cosine_results)
    index = np.where(cosine_results == min_cosine)
    moi = index[0][0] + 1
    return moi