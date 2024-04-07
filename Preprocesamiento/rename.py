import os
from PIL import Image

def rename_images(input_path, output_path, start_index=1):
    # Verifica si el directorio de salida existe, si no, créalo
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Enumera los archivos en el directorio de entrada
    for idx, filename in enumerate(os.listdir(input_path)):
        # Verifica si el archivo es una imagen JPG
        if filename.lower().endswith(('.jpg')):
            # Abre la imagen usando Pillow
            with Image.open(os.path.join(input_path, filename)) as img:
                # Construye el nuevo nombre del archivo
                new_filename = f"image{idx + start_index}.png"
                # Guarda la imagen como PNG en el directorio de salida
                img.save(os.path.join(output_path, new_filename), format='PNG')
                #print(f"Renombrando {filename} a {new_filename}")

input_path = "F:\Proyecto-SIS330\Preprocesamiento\Images"
output_path = "F:\Proyecto-SIS330\Preprocesamiento\RenameImages"
rename_images(input_path, output_path, start_index=1744)
print("Renombrado de imágenes completado...")