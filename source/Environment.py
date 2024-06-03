import cv2
import time
import torch
import numpy as np
#from RobotController.MotorController import MotorController
from Yolo.yolo_seg import YoloSeg
from CAE.maxPooling import MaxPooling

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
        for _ in range(5):
            ret, frame = self.video.read()
            time.sleep(0.1)

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
        #self.takeAction = TakeAction()
        self.yoloSeg = YoloSeg("source/Yolo/runs/segment/train3/weights/best.onnx")
        self.maxPooling = MaxPooling()
    
    def observation(self):
        frame = self.camera.getImage()
        return frame

    def step(self, action):
        #self.takeAction.action(action)
        #time.sleep(0.5)

        imgSeg = self.yoloSeg(cv2.imread("source/3.jpg"))
        print("SegImage: ", imgSeg.shape)
        imgSeg = torch.from_numpy(imgSeg).unsqueeze(0)
        imgSeg = self.maxPooling(imgSeg)
        print("MaxPooling: ", imgSeg.shape)
        observation_ = imgSeg.to(torch.float32)

        reward, done = self.calculateReward(imgSeg)

        return observation_, reward, done

    def calculateReward(self, masks):
        done = False
        # Contar los píxeles de cada tipo
        background = np.sum(masks == 0)
        obs = np.sum(masks == 1)
        camino = np.sum(masks == 2)
        print("Sum background = ", background)
        print("Sum obs = ", obs)
        print("Sum camino = ", camino)
        # Comparar las sumas y calcular la recompensa
        if camino > obs:
            reward = 1
        elif obs > camino:
            reward = -2
        else:
            reward = 0

        if camino < 2000.0:
            done = True

if __name__ == '__main__':
    env = Environment()
    observation_, reward, done = env.step(0)
    print("Observation: ", observation_[0][1][0])
    print("Reward: ", reward)
    print("Done: ", done)
