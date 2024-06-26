import cv2
import torch
import numpy as np
from UNet.UNet import UNetResnet
from PPO.Agent import Agent
from CAE.maxPooling import MaxPooling
import matplotlib.pyplot as plt
import time

if __name__ == '__main__':
    classes = ['parar', 'atras', 'adelante', 'izquierda', 'derecha', 'giroIzq', 'giroDer']
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "source/UNet/models/UNetResNet_model_seg_v3_30.pt"
    modelSegmentation = UNetResnet()
    modelSegmentation.load_model(model_path)
    modelSegmentation.to(device)
    maxPooling = MaxPooling()
    maxPooling.to(device)
    batch_size = 4
    n_epochs = 10
    alpha = 0.0003
    agent = Agent(n_actions=7, cuda=True, batch_size=batch_size, alpha=alpha, n_epochs=n_epochs)
    #agent.load_models()
    
    startTime = time.time()
    img = cv2.imread("source/6.jpg")
    print("InputImage: ", img.shape)
    x_input = torch.from_numpy(np.array(img) / 255.0).float().permute(2, 0, 1).unsqueeze(0).to(device)
    x_input = x_input.clone().detach()
    modelSegmentation.eval()
    with torch.no_grad():
        output  = modelSegmentation(x_input)[0]
        mask_img = torch.argmax(output, axis=0) #type: ignore
    seg_image = mask_img
    seg_image = seg_image.unsqueeze(0).unsqueeze(0).to(device).float()
    seg_image = maxPooling(seg_image)
    r, d = agent.calculateReward(seg_image.squeeze(0).squeeze(0).cpu().numpy())
    print("Reward: ", r, "Done: ", d)
    print("MaxPooling: ", seg_image.shape)
    inputImgPOO = seg_image
    print("InputImagePOO: ", inputImgPOO.shape, inputImgPOO.dtype, inputImgPOO.max(), inputImgPOO.min())
    action, probs, value = agent.choose_action(inputImgPOO)
    print("Action: ", action, classes[action])
    print("Probs: ", probs)
    print("Value: ", value)
    endTime = time.time()
    print("Tiempo en segundos: ", endTime - startTime)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Input Image")
    axes[0].axis('off')

    axes[1].set_title("Segmentation Image")
    axes[1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[1].imshow(mask_img.cpu().numpy(), alpha=0.5)
    axes[1].axis('off')

    inputImgPOO = inputImgPOO.squeeze(0).squeeze(0).cpu()
    axes[2].set_title("MaxPooling Image")
    axes[2].imshow(inputImgPOO)
    axes[2].axis('off')
    #print(inputImgPOO[59])
    # # Ajustar el diseño
    plt.tight_layout()
    # # Mostrar la figura
    plt.show()
    #plt.savefig('figura04.png', dpi=300, bbox_inches='tight')
    # tiempo de ejecucion es de 0.40602922439575195