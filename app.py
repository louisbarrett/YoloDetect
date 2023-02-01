# Serve Yolo as an http endpoint, and accept image/png as incoming data type
import cv2
import time
import torch
import numpy as np
from PIL import Image
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

# web server


from flask import Flask, request, jsonify
import base64
from io import BytesIO

app = Flask(__name__)

@app.route('/yolov5', methods=['POST'])
def yolov5_detection():
    model_path = "yolov5m.pt"
    received_data = request.get_json()
    # decode base64
    byte_data = base64.b64decode(received_data['image'])
    image = Image.open(BytesIO(byte_data))
    # convert to cv2
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # inferece
    start_time = time.time()
    image = yolov5_inference(image, model_path)
    end_time = time.time() - start_time
    # convert to base64
    _, img_encoded = cv2.imencode('.jpg', image)
    string_data = base64.b64encode(img_encoded).decode('utf-8')
    # return
    return jsonify({
        "image": string_data,
        "time": end_time
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)