import cv2
import torch
import numpy as np
from PPO.Environment import Environment
from Yolo.yolo_seg1 import YOLOSeg
from PPO.Agent import Agent
from CAE.maxPooling import MaxPooling
import matplotlib.pyplot as plt

def convert_to_grayscale(image):
    # Tomar la media de los dos canales para cada píxel
    grayscale_image = np.mean(image, axis=0, keepdims=True)
    return grayscale_image

def convert_to_two_channels(image):
    # Reorganizar los ejes para tener el formato (altura, ancho, canales)
    image_reordered = np.transpose(image, (1, 2, 0))
    # Seleccionar solo los dos primeros canales
    two_channel_image = image_reordered[:, :, :2]
    return two_channel_image

if __name__ == '__main__':
    env = Environment()
    modelSegmentation = YOLOSeg("source/Yolo/runs/segment/train3/weights/best.onnx")
    maxPooling = MaxPooling()
    agent = Agent(5, 1*60*108) # webcam with 3*64*108 

    #img = env.observation()
    img = cv2.imread("source/2.jpg")
    print("InputImage: ", img.shape)
    seg_image = modelSegmentation(img)
    print("SegImage: ", type(seg_image))
    seg_image = convert_to_two_channels(seg_image)
    seg_image = convert_to_grayscale(seg_image)
    print("SegImage: ", seg_image.shape)
    seg_image_torch = torch.from_numpy(seg_image).unsqueeze(0)
    print("SegImageTorch: ", seg_image_torch.shape)
    inputImgPOO = maxPooling(seg_image_torch)
    print("InputImagePOO: ", inputImgPOO.shape)
    inputImgPOO = inputImgPOO.to(torch.float32)
    action, probs, value = agent.choose_action(inputImgPOO)
    print("Action: ", action)
    print("Probs: ", probs)
    print("Value: ", value)

    seg_image = (seg_image * 255).astype(np.uint8).transpose(1, 2, 0)
    inputImgPOO=inputImgPOO.squeeze(0).permute(1, 2, 0).detach().numpy()

    while True:
        print("----------------------------------------------")
        print("InputImage: ", img.shape)
        cv2.imshow("Image Input", img)
        #seg_image = (seg_image * 255).astype(np.uint8).transpose(1, 2, 0)
        print("SegImage: ", seg_image.shape)
        cv2.imshow("Mask", seg_image)
        print("InputImagePOO: ", inputImgPOO.shape)
        cv2.imshow("InputImagePOO", inputImgPOO)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Presiona 'q' para salir
            break

    cv2.destroyAllWindows()