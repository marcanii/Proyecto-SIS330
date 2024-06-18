import socket
import cv2
import numpy as np
import threading
from Robot.MotorController import MotorController
from Agent import Agent

# Dirección IP y puerto del servidor
host_ip = '192.168.0.7'  # Dirección IP de la Jetson Nano
port = 9999

# Variables globales para compartir datos entre hilos
instruction = None
stop_threads = False

# inicializando agente
#agente = Agent()
#agente.loadModels()
motorController = MotorController()
motorController.setupMotors()

# Función para manejar la transmisión de video
def video_transmitter(client_socket):
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
        encoded, buffer = cv2.imencode('.jpg', frame)
        data = np.array(buffer)
        string_data = data.tostring()
        client_socket.sendall((str(len(string_data))).encode().ljust(512) + string_data)

    vid.release()

def handle_instruction(instruction):
    speed = 0.5
    if instruction == 0:
        motorController.stop()
        #agente.takeAction(instruction)
    elif instruction == 1:
        motorController.forward(speed)
        #agente.takeAction(instruction)
    elif instruction == 2:
        motorController.backward(speed)
        #agente.takeAction(instruction)
    elif instruction == 3:
        motorController.left(speed)
        #agente.takeAction(instruction)
    elif instruction == 4:
        motorController.right(speed)
        #agente.takeAction(instruction)
    elif instruction == 5:
        motorController.turnLeft(speed)
        #agente.takeAction(instruction)
    elif instruction == 6:
        motorController.turnRight(speed)
        #agente.takeAction(instruction)

# Función para manejar las instrucciones de control
def control_handler(client_socket):
    global instruction

    while not stop_threads:
        data = client_socket.recv(512)
        if data:
            new_instruction = int(data.decode())
            if new_instruction != instruction:
                instruction = new_instruction
                handle_instruction(instruction)

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

    video_thread.start()
    control_thread.start()

    # Esperar a que los hilos terminen
    video_thread.join()
    control_thread.join()

client_socket.close()