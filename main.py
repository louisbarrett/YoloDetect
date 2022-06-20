import cv2
import time
import torch
from transformers import YolosFeatureExtractor, YolosForObjectDetection

#Check for GPU Support 
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# init webcam
webCam = cv2.VideoCapture(0)

#Extract image features

feature_extractor = YolosFeatureExtractor.from_pretrained("hustvl/yolos-base")
#YOLO Model Small
with torch.no_grad():
    model = YolosForObjectDetection.from_pretrained("hustvl/yolos-base")
    model.to(device)
while True:
    print('\x1bc')
    return_value,image = webCam.read()
    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        inputs.to(device)
        outputs = model(**inputs)
    logits = outputs.logits
    bboxes = outputs.pred_boxes
    pred_labels = logits.argmax(dim=2)
    
    print("Detected:")
    for label in pred_labels[0]:    
        if label.item() == 91:
            continue
        print(model.config.id2label[label.item()])
    time.sleep(5)
webCam.release()
