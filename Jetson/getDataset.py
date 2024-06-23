import numpy as np
import os
from Agent import Agent
import time

agente = Agent()

def capture_and_label(label, image_dir='images'):
    img, _, _ = agente.observation()
    img = np.clip(img, 0, 2).astype(np.uint8)
    print("ShapeImg: ", img.shape, img.dtype)

    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    image_name = f"{label}_{len(os.listdir(image_dir))}.npy"
    image_path = os.path.join(image_dir, image_name)

    np.save(image_path, img)
    print(f"Imagen guardada como {image_path} y etiquetada como '{label}'")


flag = 0
while True:
    flag += 1
    capture_and_label('girar_izquierda')
    if flag == 6:
        break
    time.sleep(3)
