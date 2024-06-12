import cv2
import asyncio
import websockets
import numpy as np

# Ruta de la cámara
CAMERA_PATH = (
            "nvarguscamerasrc ! "
            "nvvidconv flip-method=0 ! "
            "video/x-raw, width=(int)864, height=(int)480, format=(string)BGRx ! "
            "videoconvert ! "
            "appsink"
        )

# Inicializar la cámara
cap = cv2.VideoCapture(CAMERA_PATH)

# Función para codificar el frame en formato jpeg
def encode_frame(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return buffer.tobytes()

# Función para enviar frames al cliente
async def send_frames(websocket, path):
    while True:
        # Leer un frame de la cámara
        ret, frame = cap.read()

        # Codificar el frame en formato jpeg
        encoded_frame = encode_frame(frame)

        # Enviar el frame al cliente
        await websocket.send(encoded_frame)

        # Esperar un corto tiempo para evitar sobrecargar el servidor
        await asyncio.sleep(0.033)  # 30 frames por segundo

# Obtener la dirección IP de la Jetson Nano
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.connect(("8.8.8.8", 80))
jetson_nano_ip = s.getsockname()[0]
s.close()

# Iniciar el servidor WebSocket
start_server = websockets.serve(send_frames, jetson_nano_ip, 8002)

# Ejecutar el servidor
print(f"Servidor WebSocket iniciado en {jetson_nano_ip}:8002")
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()