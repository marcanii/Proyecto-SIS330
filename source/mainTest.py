import cv2
import torch
import numpy as np
from Yolo.yolo_seg import YoloSeg
from UNet.UNet import UNetResnet
from PPO.Agent import Agent
from CAE.maxPooling import MaxPooling
import matplotlib.pyplot as plt
import time

if __name__ == '__main__':
    #env = Environment() F:\Proyecto-SIS330\source\Yolo\runs\segment\train4\weights\best.onnx
    #model_path = "source/Yolo/runs/segment/train3/weights/best.pt"
    #modelSegmentation = YoloSeg(model_path)
    model_path = "source/UNet/models/UNetResNet_model_seg_v3_30.pt"
    modelSegmentation = UNetResnet()
    modelSegmentation.load_model(model_path)
    maxPooling = MaxPooling()
    agent = Agent(5, cuda=True) # webcam with 3*64*108 
    
    startTime = time.time()
    #img = env.observation()
    img = cv2.imread("source/1.jpg")
    print("InputImage: ", img.shape)
    x_input = np.transpose(img, (2, 0, 1))
    x_input = torch.from_numpy(x_input).unsqueeze(0)
    x_input = x_input.to(torch.float32)
    seg_image  = modelSegmentation(x_input)
    print("SegImage: ", seg_image.shape)
    seg_image_torch = seg_image
    print("SegImageTorch: ", seg_image_torch.shape)
    inputImgPOO = maxPooling(seg_image_torch)
    print("InputImagePOO: ", inputImgPOO.shape)
    inputImgPOO = inputImgPOO.to(torch.float32)
    action, probs, value = agent.choose_action(inputImgPOO)
    print("Action: ", action)
    print("Probs: ", probs)
    print("Value: ", value)
    endTime = time.time()
    print("Tiempo en segundos: ", endTime - startTime)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Input Image")
    axes[0].axis('off')

    #seg_image = np.transpose(seg_image, (1, 2, 0))
    img_seg_show = seg_image.squeeze().detach().numpy()
    img_seg_show = np.transpose(img_seg_show, (1, 2, 0))
    img_seg_show = np.clip(img_seg_show, 0, 1)
    axes[1].set_title("Segmentation Image")
    axes[1].imshow(img_seg_show)
    axes[1].axis('off')

    axes[2].set_title("MaxPooling Image")
    inputImgPOO = inputImgPOO.squeeze().detach().numpy()
    inputImgPOO = np.transpose(inputImgPOO, (1, 2, 0))
    inputImgPOO = np.clip(inputImgPOO, 0, 1)
    axes[2].imshow(inputImgPOO)
    axes[2].axis('off')
    # # Ajustar el diseño
    plt.tight_layout()
    # # Mostrar la figura
    plt.show()
    #plt.savefig('figura04.png', dpi=300, bbox_inches='tight')