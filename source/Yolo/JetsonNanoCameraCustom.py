import cv2

# Construir la cadena de GStreamer con la altura calculada
gst_str = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080, format=(string)NV12, framerate=(fraction)30/1 ! "
    "nvvidconv flip-method=0 ! "
    "video/x-raw, width=(int)864, height=(int)480, format=(string)BGRx ! "
    "videoconvert ! "
    "appsink"
)

cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Error al abrir la cámara.")
    exit()

while True:
    ret, frame = cap.read()
    #frame = cv2.resize(frame, (1024, 512))

    if not ret:
        print("Error al leer el marco.")
        break

    cv2.imshow('Nvidia Jetson Camara IMX219-120', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
