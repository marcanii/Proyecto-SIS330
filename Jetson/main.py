import socket
import cv2
import numpy as np
import threading
from Robot.MotorController import MotorController
import requests
from io import BytesIO
import time

# URL del servidor de inferencia
url_inference = "https://dhbqf30k-5000.brs.devtunnels.ms/inference"

# Dirección IP y puerto del servidor
host_ip = '192.168.0.7'  # Dirección IP de la Jetson Nano
port = 9999

# Variables globales para compartir datos entre hilos
instruction = None
stop_threads = False
autonomous_mode = False
img = None

# Inicializando motor controller
motorController = MotorController()
motorController.setupMotors()

# Función para manejar la transmisión de video
def video_transmitter(client_socket):
    global img
    gst_str = (
        "nvarguscamerasrc ! "
        "nvvidconv flip-method=0 ! "
        "video/x-raw, width=(int)864, height=(int)480, format=(string)BGRx ! "
        "videoconvert ! "
        "appsink"
    )
    vid = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
    while vid.isOpened() and not stop_threads:
        ret, frame = vid.read()
        img = frame
        encoded, buffer = cv2.imencode('.jpg', frame)
        data = np.array(buffer)
        string_data = data.tostring()
        client_socket.sendall((str(len(string_data))).encode().ljust(128) + string_data)
    vid.release()

def inference(image):
    _, image_bytes = cv2.imencode('.jpg', image)
    image_file = BytesIO(image_bytes)
    response = requests.post(url_inference, files={'image': image_file})
    action = response.json()['action']
    return action

def handle_instruction(instruction):
    global autonomous_mode
    speed = 0.5
    if instruction == 0:
        motorController.stop()
    elif instruction == 1:
        motorController.backward(speed)
    elif instruction == 2:
        motorController.forward(speed)
    elif instruction == 3:
        motorController.left(speed)
    elif instruction == 4:
        motorController.right(speed)
    elif instruction == 5:
        motorController.turnLeft(speed)
    elif instruction == 6:
        motorController.turnRight(speed)
    elif instruction == 10:  # Cambiar a modo autónomo
        autonomous_mode = True
        print("Modo autónomo activado")
    elif instruction == 11:  # Cambiar a modo controlado
        autonomous_mode = False
        print("Modo controlado activado")

# Función para manejar las instrucciones de control
def control_handler(client_socket):
    global instruction, autonomous_mode
    while not stop_threads:
        data = client_socket.recv(128)
        if data:
            new_instruction = int(data.decode())
            if new_instruction != instruction:
                instruction = new_instruction
                handle_instruction(instruction)
        if not autonomous_mode:
            time.sleep(0.1)  # Pequeña pausa para no saturar el CPU

def autonomous_driving():
    global autonomous_mode, img
    while not stop_threads:
        if autonomous_mode and img is not None:
            action = inference(img)
            handle_instruction(action)
        else:
            time.sleep(0.1)  # Pequeña pausa para no saturar el CPU

# Configurar el servidor
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
socket_address = (host_ip, port)
server_socket.bind(socket_address)
print(f"Servidor iniciado en {host_ip}:{port}")
server_socket.listen(5)
print("Esperando conexión...")

client_socket, addr = server_socket.accept()
print(f"Conexión establecida con {addr}")

if client_socket:
    # Iniciar los hilos
    video_thread = threading.Thread(target=video_transmitter, args=(client_socket,))
    control_thread = threading.Thread(target=control_handler, args=(client_socket,))
    autonomous_thread = threading.Thread(target=autonomous_driving)
    
    video_thread.start()
    control_thread.start()
    autonomous_thread.start()
    
    # Esperar a que los hilos terminen
    video_thread.join()
    control_thread.join()
    autonomous_thread.join()

client_socket.close()