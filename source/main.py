import cv2
import torch
import numpy as np
from PPO.Environment import Environment
from Yolo.yolo_seg1 import YOLOSeg
from PPO.Agent import Agent
from CAE.maxPooling import MaxPooling
import matplotlib.pyplot as plt

if __name__ == '__main__':
    env = Environment()
    modelSegmentation = YOLOSeg("source/Yolo/runs/segment/train3/weights/best.onnx")
    maxPooling = MaxPooling()
    agent = Agent(5, 2*59*105) # webcam with 3*64*108 

    #img = env.observation()
    img = cv2.imread("source/4.jpg")
    print("InputImage: ", img.shape)
    seg_image = modelSegmentation(img)
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

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Input Image")
    axes[0].axis('off')

    seg_image = np.transpose(seg_image, (1, 2, 0))
    seg_image_gray = np.mean(seg_image, axis=2)
    axes[1].set_title("Segmentation Image")
    axes[1].imshow(seg_image_gray, cmap=None)
    axes[1].axis('off')

    axes[2].set_title("MaxPooling Image")
    inputImgPOO = inputImgPOO.squeeze(0).permute(1, 2, 0).detach().numpy()
    inputImgPOO = np.mean(inputImgPOO, axis=2)
    axes[2].imshow(inputImgPOO, cmap=None)
    axes[2].axis('off')
    # # Ajustar el diseño
    plt.tight_layout()
    # # Mostrar la figura
    plt.show()