import cv2
from Yolo.yolo_seg import YOLOSeg

if __name__ == '__main__':
    yolo = YOLOSeg("source/Yolo/runs/segment/train3/weights/best.onnx", conf_thres=0.4, iou_thres=0.4)
    video = cv2.VideoCapture(0)
    while True:
        ret, frame = video.read()
        if not ret:
            break
        seg_frame = yolo(frame)
        cv2.imshow("Segmented", seg_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video.release()
    cv2.destroyAllWindows()