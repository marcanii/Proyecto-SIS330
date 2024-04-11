import cv2

class Camera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def getImage(self):
        ret, frame = self.video.read()
        if ret:
            return frame
        else:
            print("Error al capturar la imagen.")
            return None

    def __del__(self):
        self.video.release()