import cv2
import socket
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

class VideoClient:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.client_socket = None
        self.root = tk.Tk()
        self.canvas = tk.Canvas(self.root)
        self.photo = None

    def connect(self):
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((self.host, self.port))

    def receive_and_display_video(self):
        while True:
            # Recibir el tamaño del frame
            data = self.client_socket.recv(16)
            if not data:
                break
            length = int(data)
            
            # Recibir el frame codificado
            data = b''
            while len(data) < length:
                packet = self.client_socket.recv(length - len(data))
                if not packet:
                    break
                data += packet
            
            # Decodificar el frame y mostrarlo en la ventana de Tkinter
            frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), 1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            self.photo = ImageTk.PhotoImage(image=image)

            if self.canvas.winfo_exists():
                self.canvas.delete("all")
                self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
                self.canvas.pack(fill=tk.BOTH, expand=tk.YES)
                self.root.update()

    def close(self):
        self.client_socket.close()
        self.root.destroy()

if __name__ == "__main__":
    HOST = '192.168.0.5'
    PORT = 8002

    video_client = VideoClient(HOST, PORT)
    video_client.connect()
    video_client.receive_and_display_video()
    video_client.close()
