import tkinter as tk
import socket
import cv2
import numpy as np
from PIL import Image, ImageTk

cliente_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
cliente_socket.connect(("192.168.0.5", 8002))

def getVideo(canvas):
    while True:
        # Recibir el tamaño del frame
        data = cliente_socket.recv(16)
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
            canvas.delete("all")
            canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            canvas.pack(fill=tk.BOTH, expand=tk.YES)
            root.update()

root = tk.Tk()
root.title("Controlador del Robot")

cameraFrame = tk.Frame(root)
cameraFrame.pack(side=tk.LEFT, padx=10, pady=10)
title = tk.Label(cameraFrame, text="Cámara del Robot")
title.pack()
canvas = tk.Canvas(cameraFrame, width=864, height=480)

controlFrame = tk.Frame(root)
controlFrame.pack(side=tk.LEFT, padx=10, pady=10)
title = tk.Label(controlFrame, text="Controles del Robot")
title.pack()

mode_var = tk.StringVar()
mode_var.set("autonomous")  # Modo autónomo por defecto

mode_label = tk.Label(controlFrame, text="Modo:")
mode_label.pack()

autonomous_radio = tk.Radiobutton(controlFrame, text="Autónomo", variable=mode_var, value="autonomous")
autonomous_radio.pack()

controlled_radio = tk.Radiobutton(controlFrame, text="Controlado", variable=mode_var, value="controlled")
controlled_radio.pack()

# Botones de control
forward_button = tk.Button(controlFrame, text="Adelante")
forward_button.pack(side=tk.TOP, pady=10)

backward_button = tk.Button(controlFrame, text="Atrás")
backward_button.pack(side=tk.BOTTOM, pady=10)

left_button = tk.Button(controlFrame, text="Izquierda")
left_button.pack(side=tk.LEFT, padx=3)

right_button = tk.Button(controlFrame, text="Derecha")
right_button.pack(side=tk.RIGHT, padx=3)

turn_left_button = tk.Button(controlFrame, text="Girar a la izquierda")
turn_left_button.pack(side=tk.LEFT, padx=3)

turn_right_button = tk.Button(controlFrame, text="Girar a la derecha")
turn_right_button.pack(side=tk.RIGHT, padx=3)


getVideo(canvas)

root.mainloop()