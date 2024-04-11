import cv2
from PPO.Environment import Environment
import matplotlib.pyplot as plt

if __name__ == '__main__':
    env = Environment()
    img = env.observation()

    # Convertir la imagen a RGB (si está en formato BGR)
    if len(img.shape) == 3 and img.shape[2] == 3:  # Comprueba si es una imagen a color
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Mostrar la imagen utilizando matplotlib
    plt.imshow(img)
    plt.axis('off')  # Desactiva los ejes
    plt.show()