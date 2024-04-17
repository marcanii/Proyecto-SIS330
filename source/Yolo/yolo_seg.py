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
        masks = masks.cpu().numpy()
        return self.transform_image(masks)
    
    def transform_image(self, img):
        if img.shape[0] >= 3:
            img_rgb = img[:3, :, :]
        else:
            num_channels_missing = 3 - img.shape[0]
            img_rgb = np.pad(img, ((0, num_channels_missing), (0, 0), (0, 0)), mode='constant')
        img_rgb = np.transpose(img_rgb, (1, 2, 0))
        return img_rgb

if __name__ == '__main__':
    yolo_segmentation = YoloSegmentation()
    input_image = cv2.imread("F:\Proyecto-SIS330\source\Yolo\\1.jpg")
    print("ImageInput: ", input_image.shape)
    segmented_image = yolo_segmentation.segment_image(input_image)
    print("segmented_image: ", segmented_image.shape, segmented_image.min(), segmented_image.max())
    while True:
        cv2.imshow("Mascara", segmented_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Presiona 'q' para salir
            break
    # plt.imshow(img, cmap='gray')  # Selecciona el mapa de colores 'gray' para una imagen en escala de grises
    # plt.axis('off')  # Desactivar los ejes
    # plt.show()
    cv2.destroyAllWindows()