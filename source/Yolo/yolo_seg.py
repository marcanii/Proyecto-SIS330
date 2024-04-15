import numpy as np
import onnxruntime
import cv2
import matplotlib.pyplot as plt

class YoloSegmentation:
    def __init__(self, model_path):
        self.model_path = model_path
        self.ort_session = onnxruntime.InferenceSession(model_path)
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_name = self.ort_session.get_outputs()[0].name
        print("InputShape for Model: ", self.ort_session.get_inputs()[0].shape)
        print("OutputShape for Moedel: ", self.ort_session.get_outputs()[0].shape)

    def segment_image(self, input_image):
        # Preprocesamiento de la imagen
        input_data = self._preprocess_input(input_image)
        print("InputImage for Yolo: ",input_data.shape)
        # Ejecutar la inferencia
        outputs = self.ort_session.run([self.output_name], {self.input_name: input_data})
        
        # Postprocesamiento de las salidas
        output_image = self._postprocess_output(outputs[0], input_image)

        return output_image

    def _preprocess_input(self, input_image):
        # Redimensionar la imagen de entrada al tamaño esperado por el modelo
        input_image_resized = cv2.resize(input_image, (864, 480))
        #print(input_image_resized.shape)
        input_data = np.expand_dims(input_image_resized.transpose(2, 0, 1), axis=0).astype(np.float32)
        return input_data

    def _postprocess_output(self, output_data, input_image):
        # Aplicar función de activación (por ejemplo, sigmoide) a la salida del modelo
        sigmoid_output = 1 / (1 + np.exp(-output_data))
        
        return sigmoid_output
    
    def show_segmented_image(self, segmented_image):
        plt.imshow(segmented_image)
        plt.title("Segmented Image")
        plt.axis("off")  # Ocultar los ejes
        plt.show()

if __name__ == '__main__':
    # Crear una instancia de YoloSegmentation con la ruta de tu modelo ONNX
    model_path = "F:\Proyecto-SIS330\source\Yolo\\runs\segment\\train\weights\\best.onnx"
    yolo_segmentation = YoloSegmentation(model_path)

    # Cargar la imagen de entrada
    input_image = cv2.imread("F:\Proyecto-SIS330\source\Yolo\\1.jpg")

    # Segmentar la imagen
    segmented_image = yolo_segmentation.segment_image(input_image)

    # Mostrar la imagen segmentada
    yolo_segmentation.show_segmented_image(segmented_image)
    print("ImageOuputShape: ", segmented_image.shape)
    print("ImageOuput: ", segmented_image.max(), segmented_image.min())
    cv2.waitKey(0)
    cv2.destroyAllWindows()