import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv

# leer un archivo npy
img = np.load('images/girar_izquierda_34.npy').astype(np.uint8)
print(img.shape, img.max(), img.min(), img.dtype)

plt.title('Image Forward')
plt.imshow(img)
plt.savefig('figura0.png', dpi=300, bbox_inches='tight')