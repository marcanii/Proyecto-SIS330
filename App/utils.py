import tkinter as tk
from PIL import Image, ImageTk
import cv2
import gi

gi.require_version('Gst', '1.0')
from gi.repository import Gst

# Inicializa GStreamer
Gst.init(None)

class VideoApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.video_label = tk.Label(window)
        self.video_label.pack()

        self.cap = cv2.VideoCapture('udpsrc port=5000 ! application/x-rtp, media=video, clock-rate=90000, encoding-name=H264, payload=96 ! rtph264depay ! avdec_h264 ! videoconvert ! appsink', cv2.CAP_GSTREAMER)
        
        self.update()
        self.window.mainloop()

    def update(self):
        ret, frame = self.cap.read()
        if ret:
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.video_label.configure(image=self.photo)
        self.window.after(10, self.update)

def main():
    root = tk.Tk()
    app = VideoApp(root, "Video Stream from Jetson Nano")
    root.mainloop()

if __name__ == "__main__":
    main()
