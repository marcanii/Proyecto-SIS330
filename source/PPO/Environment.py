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


class Environment:
    def __init__(self):
        self.camera = Camera()
    
    def observation(self):
        frame = self.camera.getImage()
        return frame

    def step(self, action):
        pass
