# YOLO Object Detection

This repository contains a PyTorch implementation of the YOLO object detection algorithm.

## Requirements

- Python 3.6+
- PyTorch 1.2+
- Transformers 2.8+

## Getting Started

### Installation

```bash
pip install transformers
pip install git+https://github.com/hustvl/yolos.git
```

### Usage

```python
from transformers import YolosFeatureExtractor, YolosForObjectDetection

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
```

## References

- [YOLO: Real-Time Object Detection](https://pjreddie.com/darknet/yolo/)
- [YOLOv3: An Incremental Improvement](https://pjreddie.com/media/files/papers/YOLOv3.pdf)
