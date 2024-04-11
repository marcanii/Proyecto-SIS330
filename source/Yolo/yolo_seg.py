from ultralytics import YOLO

class YoloSegmentation:
    def __init__(self):
        model = YOLO('./runs/segment/train/weights/best.pt')

    def segmentatio(self, image):
        seg = self.model(image)
        return seg