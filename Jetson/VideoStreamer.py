import cv2
import socket
import numpy as np

class VideoStreamer:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.server_socket = None
        self.conn = None
        self.addr = None
        self.cap = None
        self.frame = None
        self.gst_str = (
            "nvarguscamerasrc ! "
            "nvvidconv flip-method=0 ! "
            "video/x-raw, width=(int)864, height=(int)480, format=(string)BGRx ! "
            "videoconvert ! "
            "appsink"
        )

    def start_server(self):
        self.cap = cv2.VideoCapture(self.gst_str, cv2.CAP_GSTREAMER)
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print("Servidor iniciado")
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)  # escuchar hasta 5 conexiones

        while True:
            print("Esperando la conexión...")
            conn, addr = self.server_socket.accept()
            print(f"Conexión desde {addr}")

            try:
                while True:
                    ret, self.frame = self.cap.read()
                    encoded, buffer = cv2.imencode('.jpg', self.frame)
                    data = np.array(buffer)
                    string_data = data.tostring()
                    conn.sendall((str(len(string_data))).encode().ljust(16) + string_data)
            except (ConnectionResetError, BrokenPipeError):
                print(f"Se perdió la conexión con {addr}")
                conn.close()
            except Exception as e:
                print(f"Error: {e}")
                conn.close()

    def closeServer(self):
        print("Cerrando conexión del servidor...")
        self.server_socket.close()
        self.cap.release()
        cv2.destroyAllWindows()

    def __del__(self):
        self.closeServer()

    def getFrame(self):
        return self.frame

if __name__ == "__main__":
    HOST = "192.168.0.5"
    PORT = 8002

    video_streamer = VideoStreamer(HOST, PORT)
    video_streamer.start_server()
