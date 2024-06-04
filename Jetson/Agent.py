import requests
from io import BytesIO
import cv2
import numpy as np
from Robot.MotorController import MotorController
from Camera import Camera

class Agent:
    def __init__(self):
        #self.host, self.port = host, port
        self.url_observation = "https://dhbqf30k-5000.brs.devtunnels.ms/observation"
        self.url_chooseAction = "https://dhbqf30k-5000.brs.devtunnels.ms/chooseAction"
        self.url_remember = "https://dhbqf30k-5000.brs.devtunnels.ms/remember"
        self.url_learn = "https://dhbqf30k-5000.brs.devtunnels.ms/learn"
        self.url_saveModels = "https://dhbqf30k-5000.brs.devtunnels.ms/saveModels"
        self.url_loadModels = "https://dhbqf30k-5000.brs.devtunnels.ms/loadModels"
        self.motorController = MotorController()
        self.motorController.setupMotors()
        self.camera = Camera()

    def observation(self):
        image = self.camera.getImage()
        _, image_bytes = cv2.imencode('.jpg', image)
        image_file = BytesIO(image_bytes)
        response = requests.post(self.url_observation, files={'image': image_file})
        img = response.json()['image']
        reward = response.json()['reward']
        done = response.json()['done']
        observation_ = np.array(img)
        return observation_, reward, done

    def chooseAction(self, imgSeg):
        imgSeg = imgSeg.tolist()
        #print("ShapeImage:", imgSeg.shape, imgSeg.dtype)
        response = requests.post(self.url_chooseAction, json={'image': imgSeg})
        return response.json()['action'], response.json()['probs'], response.json()['value']
    
    def remember(self, state, action, probs, value, reward, done):
        #print("State: ", type(state))
        state = np.expand_dims(state, axis=0).tolist()
        response = requests.post(self.url_remember, json={'state': state, 'action': action, 'probs': probs, 'value': value, 'reward':reward, 'done': done})
        return response.status_code
    
    def learn(self):
        response = requests.post(self.url_learn)
        return response.status_code
    
    def step(self, action):
        self.takeAction(action)
        observation_, reward, done = self.observation()
        #reward, done = self.calculateReward(imgSeg)
        #observation_ = imgSeg
        return observation_, reward, done
    
    def calculateReward(self, masks):
        done = False
        # Contar los píxeles de cada tipo
        background = np.sum(masks == 0)
        obs = np.sum(masks == 1)
        camino = np.sum(masks == 2)
        # print("Sum background = ", background)
        # print("Sum obs = ", obs)
        # print("Sum camino = ", camino)
        # Comparar las sumas y calcular la recompensa
        if camino > obs:
            reward = 1
        elif obs > camino:
            reward = -2
            done = True
        else:
            reward = 0
            done = True

        return reward, done
    
    def getDone(self, reward):
        if reward < -10:
            return True
        return False


    def takeAction(self, action):
        if action == 0:
            self.motorController.stop()
        elif action == 1:
            self.motorController.backward(speed=0.4)
        elif action == 2:
            self.motorController.forward(speed=0.4)
        elif action == 3:
            self.motorController.left(speed=0.4)
        elif action == 4:
            self.motorController.right(speed=0.4)
        elif action == 5:
            self.motorController.turnLeft(speed=0.4)
        elif action == 6:
            self.motorController.turnRight(speed=0.4)

    def __del__(self):
        self.motorController.stop()

    def saveModels(self):
        response = requests.post(self.url_saveModels, json={'save': True})
        if response.status_code == 200:
            return True
        
        return False
    
    def loadModels(self):
        response = requests.post(self.url_loadModels, json={'load': True})
        if response.status_code == 200:
            return True
        
        return False