import cv2
import numpy as np
import os 
import argparse

from keras.models import model_from_json
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.mobilenet_v2 import preprocess_input
from sklearn.preprocessing import LabelEncoder

# from utils.parser import get_config
def load_model_clf(cfg):
    json_file = open(cfg.mobileNet.JSON)
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(cfg.mobileNet.WEIGHTS)
    labels = LabelEncoder()
    labels.classes_ = np.load(cfg.mobileNet.CLASSES)
    return model, labels

def predict_from_model(image, model, labels):
    image = cv2.resize(image,(80,80))
    # image = np.stack((image,)*3, axis=-1)
    prediction = labels.inverse_transform([np.argmax(model.predict(image[np.newaxis,:]))])
    return prediction

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_classifier", type=str, default="./configs/mobileNet.yaml")
    return parser.parse_args()

if __name__ == '__main__':
    cfg = get_config()
    args = parse_args()
    cfg.merge_from_file(args.config_classifier)

    # class 12
    image = cv2.imread("demo_clf.jpg")

    # load model
    model, labels = load_model_clf(cfg)

    # predict = np.array2string(predict_from_model(image, model, labels))
    predict = predict_from_model(image, model, labels)
    print("predict: ", type(predict))