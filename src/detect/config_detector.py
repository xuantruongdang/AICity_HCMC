import cv2
import os
import numpy as np

def config_yolov3(cfg):
    net = cv2.dnn.readNet(cfg.YOLOV3.WEIGHTS, cfg.YOLOV3.CFG)

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return net, layer_names, output_layers