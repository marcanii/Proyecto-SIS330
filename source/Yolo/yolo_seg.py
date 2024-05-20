from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt

class YoloSeg:
    def __init__(self, path):
        self.model = YOLO(path)

    def __call__(self, image):
        self.width = image.shape[1]
        self.height = image.shape[0]
        return self.segmentation(image)
    
    def segmentation(self, image):
        predicts = self.model.predict(image)[0]
        masks = predicts.masks.data
        return self.binary_mask(masks.cpu().numpy())

    def binary_mask(self, mask_maps):
        mask_maps = np.array(mask_maps)
        # Obtener la dimensión de la primera dimensión
        n = mask_maps.shape[0]
        print("Masks: ", n)
        if n == 0:
            return np.zeros((2, self.height, self.width))
        # Crear una nueva matriz de dimensión (2, 480, 864) inicializada con ceros
        reshaped_output = np.zeros((2, self.height, self.width))

        # Caso 1: Si la entrada es de un solo canal
        if n == 1:
            reshaped_output[0] = mask_maps[0]
        # Caso 2: Si la entrada tiene más de un canal
        else:
            reshaped_output[0] = mask_maps[0]
            # Si la dimensión original es mayor que 2, calcular la media de los canales restantes y asignarla al segundo canal
            if n > 2:
                reshaped_output[1] = np.mean(mask_maps[2:], axis=0)
            else:
                reshaped_output[1] = mask_maps[1]
        
        return reshaped_output  

if __name__ == '__main__':
    model_path = "source/Yolo/runs/segment/train3/weights/best.pt"
    # Initialize YOLOv8 Instance Segmentator
    yoloseg = YoloSeg(model_path)

    img = cv2.imread("source/3.jpg")
    print("Input Shape: ", img.shape)
    x = yoloseg(img)
    print("Output Shape: ", x.shape)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Input Image")
    axes[0].axis('off')

    axes[1].set_title("Segmentation Image")
    axes[1].imshow(x[0], cmap=None)
    axes[1].axis('off')
    # # Ajustar el diseño
    plt.tight_layout()
    # # Mostrar la figura
    plt.show()