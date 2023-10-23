import cv2
import urllib.request
from flask import Flask, Response, render_template
import numpy as np

app = Flask(__name__)

# Model detection
classFile = 'coco.names'
classNames = []
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

##########CAMARAS######
url1 = 'http://192.168.0.100/cam-lo.jpg'
url2 = 'http://192.168.0.101/cam-lo.jpg'

# Funcion para detectar personas y generar frames
def generate_frames(url):
    net = cv2.dnn_DetectionModel(weightsPath, configPath)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)
    
    while True:
        imgResponse = urllib.request.urlopen(url)
        imgNp = np.array(bytearray(imgResponse.read()), dtype=np.uint8)
        frame = cv2.imdecode(imgNp, -1)
        classIds, confs, bbox = net.detect(frame, confThreshold=0.5)
        print(classIds, bbox)
        if len(classIds) != 0:
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                if classId == 1:
                    cv2.rectangle(frame, box, color=(0, 255, 0), thickness=3)
                    cv2.putText(frame, classNames[classId - 1], (box[0] + 10, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        ret, buffer1 = cv2.imencode('.jpg', frame)
        if not ret:
            break
        frame_bytes1 = buffer1.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes1 + b'\r\n')
    
    # Liberar referencias y recursos del modelo
    net.clear()
    cv2.destroyAllWindows()

Cam1 = generate_frames(url1)
Cam2 = generate_frames(url2)

@app.route('/')
def home():
    return render_template("SmartLight.html")

@app.route('/video_feed1')
def video_feed1():
    return Response(Cam1, mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed2')
def video_feed2():
    return Response(Cam2, mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host="192.168.0.102", port=8000)
