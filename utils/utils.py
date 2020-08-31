import numpy as np
import cv2
import os 

from shapely.geometry import Point, Polygon

def config_cam(img, cfg, args):
    line = cfg.CAM.LINE
    line_startX = []
    line_endX = []
    line_startY = []
    line_endY = []

    moi = cfg.CAM.MOI
    moi_startX = []
    moi_endX = []
    moi_startY = []
    moi_endY = []
    
    roi_default = np.array(cfg.CAM.ROI_DEFAULT)
    # tracking_roi = np.array(cfg.CAM.TRACKING_ROI)
    roi_btc = np.array(cfg.CAM.ROI_BTC)

    color_list = [(255,0,255), (255,100,0), (0,255,0), (139, 69, 19), (132, 112, 255), (0, 154, 205), (0, 255, 127), (238, 180, 180),
                  (255, 69, 0), (238, 106, 167), (221, 160, 221), (0, 128, 128)]

    # plot ROI
    cv2.drawContours(img, [roi_default], -1, (0, 0, 255), 2)
    # cv2.drawContours(img, [tracking_roi], -1, (0, 255, 0), 2)
    cv2.drawContours(img, [roi_btc], -1, (255, 0, 0), 2)

    # plot calibrations lines for count by line
    if args.count == "line":
        for i in line:
            line_startX.append (i[0][0])
            line_startY.append (i[0][1])
            line_endX.append (i[1][0])
            line_endY.append (i[1][1])
          
        for i in range (len(line_startX)):
            cv2.line(img, (line_startX[i], line_startY[i]), (line_endX[i], line_endY[i]),color_list[i])

    # plot MOI
    for i in moi:
        moi_startX.append (i[0][0])
        moi_startY.append (i[0][1])
        moi_endX.append (i[1][0])
        moi_endY.append (i[1][1])
    
    for i in range (len(moi_startX)):
        cv2.arrowedLine(img, (moi_startX[i], moi_startY[i]), (moi_endX[i], moi_endY[i]), color_list[i], thickness=2, tipLength=0.03)

    return img
    

def init_board(image, number_MOI=6, col=70, row1=45, row2=65, row3=85, row4=105):
    list_col = []
    cv2.putText(image, "Loai_1:", (5, row1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
    cv2.putText(image, "Loai_2:", (5, row2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    cv2.putText(image, "Loai_3:", (5, row3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    cv2.putText(image, "Loai_4:", (5, row4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    i = 1
    while i <= number_MOI:
        cv2.putText(image, "MOI_{}".format(i), (col, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        list_col.append(col)

        col += 50
        i += 1

    return image, list_col


def write_board(image, arr_cnt, list_col, number_MOI=6, row1=45, row2=65, row3=85, row4=105):
    list_row = [row1, row2, row3, row4]
    list_color = [(255, 0, 255), (0, 0, 255), (255, 0, 0), (0, 255, 0)]
    col = 1
  
    while col <= number_MOI:
        row = 1
        while row<=4:
            cv2.putText(image, "{}".format(arr_cnt[row-1][col-1]), (list_col[col-1]+20, list_row[row-1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, list_color[row-1], 1)
            row += 1 
        col += 1

    return image

'''
    Check obj in polygon
    input: list vertices
    output: True if in polygon, otherwise
'''
def check_in_polygon(center_point, polygon):
    pts = Point(center_point[0], center_point[1])
    if polygon.contains(pts):
        return True
    
    return False

'''
    return mask(ROI) of MOI
'''
def check_number_MOI(number, cfg):
    number_MOI = cfg.CAM.NUMBER_MOI
    if number_MOI == 2:
        switcher = {
            1: Polygon(cfg.CAM.ROI1),
            2: Polygon(cfg.CAM.ROI2)
        }
    if number_MOI == 6:
        switcher = {
            1: Polygon(cfg.CAM.ROI1),
            2: Polygon(cfg.CAM.ROI2),
            3: Polygon(cfg.CAM.ROI3),
            4: Polygon(cfg.CAM.ROI4), 
            5: Polygon(cfg.CAM.ROI5),
            6: Polygon(cfg.CAM.ROI6)
        }

    return switcher.get(number, "Invalid ROI of cam")