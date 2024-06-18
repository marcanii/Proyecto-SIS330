import tkinter as tk
from tkinter import ttk
import tkinter as tk
from tkinter import ttk
import socket
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading

# Conectar al servidor
cliente_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
cliente_socket.connect(("192.168.0.7", 9999))
accion = None

def getVideo(canvas, image_label):
    while True:
        # Recibir el tamaño del frame
        data = cliente_socket.recv(512)
        if not data:
            break
        length = int(data)
        
        # Recibir el frame codificado
        data = b''
        while len(data) < length:
            packet = cliente_socket.recv(length - len(data))
            if not packet:
                break
            data += packet
        
        # Decodificar el frame y mostrarlo en la ventana de Tkinter
        frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        photo = ImageTk.PhotoImage(image=image)

        if canvas.winfo_exists():
            image_label.configure(image=photo)
            image_label.image = photo
            root.update()

def send_instruction(instruction):
    cliente_socket.send(str(instruction).encode())

def move_backward():
    send_instruction(1)

def move_forward():
    send_instruction(2)

def move_left():
    send_instruction(3)

def move_right():
    send_instruction(4)

def turn_left():
    send_instruction(5)

def turn_right():
    send_instruction(6)

def stop_robot():
    send_instruction(0)

def close_app():
    cliente_socket.close()
    root.destroy()

def set_mode():
    if mode_var.get() == "controlled":
        enable_controls()
    else:
        disable_controls()
        send_instruction(10)

def enable_controls():
    forward_button.config(state=tk.NORMAL)
    left_button.config(state=tk.NORMAL)
    right_button.config(state=tk.NORMAL)
    backward_button.config(state=tk.NORMAL)
    turn_left_button.config(state=tk.NORMAL)
    turn_right_button.config(state=tk.NORMAL)
    stop_button.config(state=tk.NORMAL)

def disable_controls():
    forward_button.config(state=tk.DISABLED)
    left_button.config(state=tk.DISABLED)
    right_button.config(state=tk.DISABLED)
    backward_button.config(state=tk.DISABLED)
    turn_left_button.config(state=tk.DISABLED)
    turn_right_button.config(state=tk.DISABLED)
    stop_button.config(state=tk.DISABLED)

root = tk.Tk()
root.title("Controlador del Robot")

# Estilos
style = ttk.Style()
style.configure("TFrame", background="lightgray")
style.configure("TButton", padding=6)
style.configure("TRadiobutton", background="lightgray")

# Frame de la cámara
camera_frame = ttk.Frame(root, padding="10 10 10 10")
camera_frame.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

camera_title = ttk.Label(camera_frame, text="Cámara del Robot", font=("Helvetica", 16))
camera_title.grid(column=0, row=0, pady=10)

canvas = tk.Canvas(camera_frame, width=864, height=480)
canvas.grid(column=0, row=1)

image_label = ttk.Label(camera_frame)
image_label.grid(column=0, row=1)

# Frame de controles
control_frame = ttk.Frame(root, padding="10 10 10 10")
control_frame.grid(column=1, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))
root.columnconfigure(1, weight=1)
root.rowconfigure(0, weight=1)

control_title = ttk.Label(control_frame, text="Controles del Robot", font=("Helvetica", 16))
control_title.grid(column=0, row=0, columnspan=2, pady=10)

mode_var = tk.StringVar()
mode_var.set("controlled")  # Modo autónomo por defecto

mode_label = ttk.Label(control_frame, text="Modo:")
mode_label.grid(column=0, row=1, sticky=tk.W, pady=5)

autonomous_radio = ttk.Radiobutton(control_frame, text="Autónomo", variable=mode_var, value="autonomous", command=set_mode)
autonomous_radio.grid(column=0, row=2, sticky=tk.W)

controlled_radio = ttk.Radiobutton(control_frame, text="Controlado", variable=mode_var, value="controlled", command=set_mode)
controlled_radio.grid(column=1, row=2, sticky=tk.W)

# Botones de control
forward_button = ttk.Button(control_frame, text="Adelante", command=move_forward)
forward_button.grid(column=0, row=3, columnspan=2, pady=5)

left_button = ttk.Button(control_frame, text="Izquierda", command=move_left)
left_button.grid(column=0, row=4, pady=5, sticky=tk.W)

right_button = ttk.Button(control_frame, text="Derecha", command=move_right)
right_button.grid(column=1, row=4, pady=5, sticky=tk.E)

backward_button = ttk.Button(control_frame, text="Atrás", command=move_backward)
backward_button.grid(column=0, row=5, columnspan=2, pady=5)

turn_left_button = ttk.Button(control_frame, text="Girar a la izquierda", command=turn_left)
turn_left_button.grid(column=0, row=6, pady=5, sticky=tk.W)

turn_right_button = ttk.Button(control_frame, text="Girar a la derecha", command=turn_right)
turn_right_button.grid(column=1, row=6, pady=5, sticky=tk.E)

# Botones adicionales
stop_button = ttk.Button(control_frame, text="Detener", command=stop_robot)
stop_button.grid(column=0, row=7, columnspan=2, pady=10)

close_button = ttk.Button(control_frame, text="Cerrar", command=close_app)
close_button.grid(column=0, row=8, columnspan=2, pady=10)

# Iniciar la captura de video
#getVideo(canvas, image_label)

# Iniciar el hilo para la captura de video
video_thread = threading.Thread(target=getVideo, args=(canvas, image_label))
video_thread.start()

root.mainloop()