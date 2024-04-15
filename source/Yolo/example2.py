import onnxruntime as ort
import cv2
import numpy as np

# Cargar el modelo ONNX
model = ort.InferenceSession("F:\Proyecto-SIS330\source\Yolo\\runs\segment\\train\weights\\best.onnx")

# Obtener los nombres de las entradas y salidas del modelo
input_names = model.get_inputs()
output_names = model.get_outputs()

# Definir los nombres de las entradas y salidas
input_tensor_name = input_names[0].name
output_tensor_names = output_names

#print(input_tensor_name, output_tensor_names)

# Leer la imagen
image = cv2.imread("F:\Proyecto-SIS330\source\Yolo\\1.jpg")

# Convertir la imagen a BGR (OpenCV usa BGR)
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

image = cv2.resize(image, (864, 480))

# Normalizar la imagen entre 0 y 1
image = image / 255.0

# Transponer la imagen de (H, W, C) a (C, H, W)
image = image.transpose((2, 0, 1))

# Expandir la dimensión de la imagen a (1, C, H, W)
image = np.expand_dims(image, axis=0)
image = image.astype(np.float32)
# Realizar la inferencia del modelo
detections = model.run(["output1"], {"images": image})
#print(detections[0].shape)

heatmaps = detections[0][0].transpose((1, 2, 0))
print("heatmaps: ",heatmaps.shape)
threshold = 0.5
boxes = (heatmaps > threshold).astype(np.uint8)
# Asegúrate de que boxes es una matriz de tipo np.uint8 con valores binarios
boxes = (boxes > threshold).astype(np.uint8) * 255

# Verifica el tipo y la forma de boxes
print(f"boxes.dtype: {boxes.dtype}")
print(f"boxes.shape: {boxes.shape}")
print("min: ",boxes.min(), "max: ", boxes.max())

# Encuentra los contornos en boxes
contours, _ = cv2.findContours(boxes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
output = np.zeros((2, 480, 864))
colors = [(0, 0, 255), (0, 255, 0)]  # define un color para cada clase de objeto
for i, contour in enumerate(contours):
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(output[0], (x, y), (x+w, y+h), colors[i % len(colors)], 2)

cv2.imshow('Output', output[0])
cv2.waitKey(0)
cv2.destroyAllWindows()
#cv2.imwrite('output.png', output[0])