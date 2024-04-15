import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar el modelo YOLOv5
net = cv2.dnn.readNet("F:\Proyecto-SIS330\source\Yolo\\runs\segment\\train\weights\\best.onnx")

# Leer la imagen de entrada
image_path = "F:\Proyecto-SIS330\source\Yolo\\1.jpg"
image = cv2.imread(image_path)

# Convertir la imagen a RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Crear un blob a partir de la imagen
blob = cv2.dnn.blobFromImage(image, 1/255.0, (864, 480), swapRB=True, crop=False)

# Establecer la entrada del modelo
net.setInput(blob)

# Obtener la salida del modelo
output = net.forward()

# Obtener la forma de la salida
output_shape = output.shape

# Extraer las dimensiones de la imagen original
original_image_height, original_image_width = image.shape[:2]

# Inicializar una máscara vacía
mask = np.zeros((original_image_height, original_image_width), dtype=np.uint8)

# Recorrer las predicciones de cada clase
for class_index in range(output_shape[1]):
    # Obtener las predicciones para la clase actual
    class_predictions = output[0, class_index]

    # Recorrer las detecciones de la clase actual
    for detection in class_predictions:
        # Obtener la puntuación de confianza de la detección
        confidence = detection[5]

        # Filtrar detecciones con baja confianza
        if confidence < 0.5:
            continue

        # Obtener las coordenadas del cuadro delimitador
        xmin, ymin, xmax, ymax = detection[:4] * original_image_width, detection[:4] * original_image_height

        # Extraer la máscara de la detección
        detection_mask = output[0, class_index + 6:class_index + 6 + 120 * 216].reshape((216, 120))

        # Redimensionar la máscara de la detección a la imagen original
        resized_mask = cv2.resize(detection_mask, dsize=(int(xmax - xmin), int(ymax - ymin)))

        # Superponer la máscara de la detección a la máscara final
        mask[int(ymin):int(ymax), int(xmin):int(xmax)] = np.maximum(mask[int(ymin):int(ymax), int(xmin):int(xmax)], resized_mask)

# Convertir la máscara a escala de grises
mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

# Mostrar la imagen y la máscara
plt.subplot(121), plt.imshow(image), plt.title('Imagen original')
plt.subplot(122), plt.imshow(mask), plt.title('Máscara segmentada')
plt.show()
# a = output[0][0]
# # aplica sigmoid 
# sigmoid_a = 1 / (1 + np.exp(-a))
# binary_a = np.where(sigmoid_a > 0.5, 1, 0)
# plt.figure(figsize=(8, 6))
# plt.imshow(binary_a, cmap='gray')
# plt.axis('off')
# plt.title("Salida binarizada")
# plt.show()

# # Imprime información
# print("Forma de la salida binarizada:", binary_a.shape)
# print("Valor mínimo:", binary_a.min())
# print("Valor máximo:", binary_a.max())