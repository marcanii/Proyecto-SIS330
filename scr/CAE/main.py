from AutoEncode import AutoEncode
import torch
import matplotlib.pyplot as plt
import numpy as np
from maxPooling import *
from PIL import Image

def plot_imgs(img_pil, out):
    fig, ax = plt.subplots(1, 2, figsize=(15, 10))  # 1 fila, 2 columnas, tamaño de la figura ajustable
    # Mostrar la imagen original
    ax[0].imshow(img_pil)
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    # Mostrar la imagen reconstruida
    out_np = out.squeeze(0).permute(1, 2, 0).detach().numpy()
    ax[1].imshow(out_np)
    ax[1].set_title('Reconstructed Image')
    ax[1].axis('off')

    plt.show()

if __name__ == '__main__':
    img_path = 'E:\\7°Semestre\Inteligencia Artificial III\Proyecto-SIS330\scr\CAE\\1.jpg'
    img_pil = Image.open(img_path).convert('RGB')
    #plt.imshow(img_pil)
    #plt.show()
    img = np.array(img_pil)
    img = img / 255
    img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
    print(img.shape)
    model = ConvolutionalMaxPooling()
    out = model(img)
    print(out.shape)
    plot_imgs(img_pil, out)