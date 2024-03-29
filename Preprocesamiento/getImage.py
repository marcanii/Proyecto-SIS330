import cv2
import os
import numpy as np

def extract_images(video_path, output_folder, frame_rate=1):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    success, frame = video.read()
    index = 1785
    new_width = 840
    new_height = int((840 / 1920) * 1080)

    while success:
        if index % (fps * frame_rate) == 0:
            resized_frame = cv2.resize(frame, (new_width, new_height))
            file_name = f"frame_{index}.jpg"
            file_path = os.path.join(output_folder, file_name)
            cv2.imwrite(file_path, resized_frame)

        success, frame = video.read()
        index += 1

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "F:\Proyecto-SIS330\Preprocesamiento\Videos\\3.mp4"  # Reemplaza con la ruta a tu archivo de video
    output_folder = "./Images"  # Reemplaza con la ruta a la carpeta de salida
    frame_rate = 1  # Extrae una imagen por segundo

    extract_images(video_path, output_folder, frame_rate)
    print("Extracción de imágenes completada...")