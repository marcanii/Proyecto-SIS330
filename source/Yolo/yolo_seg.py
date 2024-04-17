import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np

class YoloSegmentation:
    def __init__(self):
        self.model = YOLO('F:\Proyecto-SIS330\source\Yolo\\runs\segment\\train3\weights\\best.pt')

    def segment_image(self, input_image):
        # Realiza la predicción
        predictions = self.model.predict(input_image)[0]
        masks = predictions.masks.data
        return masks.cpu().numpy()

if __name__ == '__main__':
    yolo_segmentation = YoloSegmentation()
    input_image = cv2.imread("F:\Proyecto-SIS330\source\Yolo\\2.jpg")
    print("ImageInput: ", input_image.shape)
    segmented_image = yolo_segmentation.segment_image(input_image)
    print("segmented_image: ", segmented_image.shape, segmented_image.min(), segmented_image.max())
    img = segmented_image[0]
    plt.imshow(img, cmap='gray')  # Selecciona el mapa de colores 'gray' para una imagen en escala de grises
    plt.axis('off')  # Desactivar los ejes
    plt.show()
    cv2.destroyAllWindows()