import cv2
import torch
import numpy as np
from PPO.Environment import Environment
from Yolo.yolo_seg import YoloSegmentation
from PPO.Agent import Agent
from CAE.maxPooling import MaxPooling
import matplotlib.pyplot as plt

if __name__ == '__main__':
    env = Environment()
    modelSegmentation = YoloSegmentation()
    maxPooling = MaxPooling()
    agent = Agent(5, 3*60*108) # webcam with 3*64*108 

    #img = env.observation()
    img = cv2.imread("F:\Proyecto-SIS330\source\\2.jpg")
    print("InputImage: ", img.shape)
    seg_image = modelSegmentation.segment_image(img)
    print("SegImage: ", seg_image.shape)
    seg_image_torch = np.transpose(seg_image, (2, 0, 1))
    seg_image_torch = torch.from_numpy(seg_image_torch).unsqueeze(0)
    print("SegImageTorch: ", seg_image_torch.shape)
    inputImgPOO = maxPooling(seg_image_torch)
    print("InputImagePOO: ", inputImgPOO.shape)
    action, probs, value = agent.choose_action(inputImgPOO)
    print("Action: ", action)
    print("Probs: ", probs)
    print("Value: ", value)

    while True:
        cv2.imshow("Image Input", img)
        cv2.imshow("Mask", seg_image)
        cv2.imshow("InputImagePOO", inputImgPOO.squeeze(0).permute(1, 2, 0).detach().numpy())
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Presiona 'q' para salir
            break

    cv2.destroyAllWindows()