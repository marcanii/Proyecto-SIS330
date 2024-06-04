import requests
import numpy as np
from Camera import Camera
import cv2
from io import BytesIO
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from Agent import Agent

url_observation = "https://dhbqf30k-5000.brs.devtunnels.ms/observation"
camera = Camera()
image = camera.getImage()
#image = cv2.imread("1.jpg")
_, image_bytes = cv2.imencode('.jpg', image)
image_file = BytesIO(image_bytes)

# Enviar la solicitud POST con la imagen
response = requests.post(url_observation, files={'image': image_file})
segmented_image_data = response.json()['image']
img_ = np.array(segmented_image_data)
print("ShapeImage:", img_.shape, img_.dtype, img_.max(), img_.min())
print(img_[300])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))
ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
ax2.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
ax2.imshow(img_, alpha=0.5)

plt.tight_layout()
# # Mostrar la figura
plt.savefig('figura02.png', dpi=300, bbox_inches='tight')