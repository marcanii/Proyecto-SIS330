import cv2

gst_str = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080, format=(string)NV12, framerate=(fraction)30/1 ! "
    "nvvidconv flip-method=0 ! "
    "video/x-raw, width=(int)864, height=(int)480, format=(string)BGRx ! "
    "videoconvert ! "
    "appsink"

)

class Camera:
    def __init__(self):
        self.video = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

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
