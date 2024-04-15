from ultralytics import YOLO
import numpy as np
import cv2

def preprocess_input(image):
    # Resize image to match model input shape
    resized_image = cv2.resize(image, (864, 480))
    # Normalize pixel values to range [0, 1]
    normalized_image = resized_image.astype(np.float32) / 255.0
    # Transpose image to match model input shape [B, C, H, W]
    input_image = np.transpose(normalized_image, (2, 0, 1))
    # Add batch dimension
    input_image = np.expand_dims(input_image, axis=0)
    return resized_image

# Load a pretrained YOLOv8n-seg Segment model
model = YOLO('F:\Proyecto-SIS330\source\Yolo\\runs\segment\\train\weights\\best.onnx')
image = cv2.imread("F:\Proyecto-SIS330\source\Yolo\\1.jpg")
a = preprocess_input(image)
print(a.shape)
results = model.predict(source=a, save=True)
for r in results:
    print(r.masks.shape)