import cv2
import time
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision.transforms import ToPILImage
import yolov5
# medium model
# https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5m.pt

def yolov5_inference(
    image = None,
    model_path = None,
    image_size = 640,
    conf_threshold = 0.25,
    iou_threshold = 0.45,
):
    """
    YOLOv5 inference function
    Args:
        image: Input image
        model_path: Path to the model
        image_size: Image size
        conf_threshold: Confidence threshold
        iou_threshold: IOU threshold
    Returns:
        Rendered image
    """
    model = yolov5.load(model_path, device="cpu")
    model.conf = conf_threshold
    model.iou = iou_threshold
    results = model([image], size=image_size)
    return results.render()[0]
        

#Check for GPU Support 
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# init webcam
webCam = cv2.VideoCapture(0)

#Extract image features

while True:
    # print('\x1bc')
    return_value,image = webCam.read()
    annotated_image = yolov5_inference(image, "./yolov5s.pt")
    cv2.imshow('webcam',annotated_image)
    if cv2.waitKey(1)& 0xFF == ord('q'):
        break
    time.sleep(0.01)
  
webCam.release()
