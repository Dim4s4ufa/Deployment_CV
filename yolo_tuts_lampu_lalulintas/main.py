from flask import Flask, render_template, request, Response
from ultralytics import YOLO
from PIL import Image
import cv2
import threading

streaming_event = threading.Event()

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def home():
  if request.method == 'GET':
    return render_template("index.html")

  elif request.method == 'POST':
    f = request.files['img']
    img = Image.open(f.stream)
    model = YOLO("models/best-v2.pt")
    
    # Prediksi gambar
    result = model.predict(img)
    result[0].save("static/predicted.jpg")
    
    return render_template("index.html", predicted=True)
  
@app.route("/video_stream")
def video_stream():
  return Response(video_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/realtime-object-detection")
def realtime_obj_detection():
  global streaming_event
  streaming_event.set()
  return render_template("stream.html", on_air=streaming_event)

@app.route("/realtime-object-detection/stop")
def stop_streaming():
  global streaming_event
  streaming_event.clear()
  return "streaming berhenti"

@app.route("/video_stream")
def video_frame():
  # Source: web cam
  cap = cv2.VideoCapture(0)
  
  while(cap.isOpened()):
    success, frame = cap.read()
    if not success:
      break
    
    global streaming_event
    if not streaming_event.is_set():
      break
    
    model = YOLO("models/best-v2.pt")
    result = model(frame)
    
    img_byte = result[0].plot()
    frame_bytes = cv2.imencode('.jpg', img_byte)[1].tobytes() 
    
    yield (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

if __name__ == '__main__':
  app.run(port=5001, debug=True)