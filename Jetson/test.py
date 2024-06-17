import requests
import numpy as np
import cv2
from io import BytesIO
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from Agent import Agent
from Camera import Camera

url_observation = "https://dhbqf30k-5000.brs.devtunnels.ms/observation"
url_chooseAction = "https://dhbqf30k-5000.brs.devtunnels.ms/chooseAction"
camera = Camera()
#agent = Agent()

image = camera.getImage()
#image = cv2.imread("1.jpg")
#time.sleep(2)
#image, _, _ = agent.observation()
# Codificar la imagen como bytes
_, image_bytes = cv2.imencode('.jpg', image)
image_file = BytesIO(image_bytes)

# Enviar la solicitud POST con la imagen
response = requests.post(url_observation, files={'image': image_file})

if response.status_code == 200:
    inicio = time.time()
    # Obtener los datos de la imagen segmentada de la respuesta JSON
    segmented_image_data = response.json()['image']
    reward = response.json()['reward']
    done = response.json()['done']
    # Convertir la lista a una matriz NumPy
    img_ = np.array(segmented_image_data)
    print("ShapeImage:", img_.shape, img_.dtype)
    print("Reward:", reward, "Done:", done)
    img = img_.tolist()
    response = requests.post(url_chooseAction, json={'image': img})
    action = response.json()['action']
    probs = response.json()['probs']
    value = response.json()['value']
    print("Time:", time.time()-inicio)
    print("Action:", action, "Probs:", probs, "Value:", value)
    
else:
    print(f'Error: {response.status_code} - {response.text}')

fig, axes = plt.subplots(1, 2, figsize=(16, 4))
axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0].set_title("Input Image")
axes[0].axis('off')

axes[1].set_title("Segmentation Image")
axes[1].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[1].imshow(img_, alpha=1.0)
axes[1].axis('off')

print(img_.max(), img_.min())
#print("Tiempo espera...")
#time.sleep(10)
# # Ajustar el diseño
plt.tight_layout()
# # Mostrar la figura
plt.savefig('figura09.png', dpi=300, bbox_inches='tight')
print("Done")
# tiempo de ejecucion 1.5405511856079102