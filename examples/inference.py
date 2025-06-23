#!/usr/bin/env python3
import io
import time
from typing import List, Tuple

import cv2
import numpy as np
from flask import Flask, Response

# TFLite Interpreter import (경량화된 tflite_runtime 우선, 없으면 TensorFlow)
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

from picamera import PiCamera
from picamera.array import PiRGBArray

app = Flask(__name__)

@app.route('/')
def index():
    return """
    <html>
      <head><title>Fire Detection Stream</title></head>
      <body>
        <h3>실시간 화재 감지 스트림</h3>
        <img src="/video_feed" style="width:100%; height:auto;">
      </body>
    </html>
    """

def run_inference(interpreter: Interpreter, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    h_input, w_input = input_details[0]["shape"][1:3]
    resized = cv2.resize(image, (w_input, h_input))
    input_data = np.expand_dims(resized, axis=0).astype(np.uint8)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    scores  = interpreter.get_tensor(output_details[0]['index'])[0]
    boxes   = interpreter.get_tensor(output_details[1]['index'])[0]
    count   = int(interpreter.get_tensor(output_details[2]['index'])[0])
    classes = interpreter.get_tensor(output_details[3]['index'])[0].astype(int)

    return boxes[:count], classes[:count], scores[:count]

def draw_predictions(
    image: np.ndarray,
    boxes: np.ndarray,
    classes: np.ndarray,
    scores: np.ndarray,
    class_names: List[str],
    threshold: float = 0.5,
) -> np.ndarray:
    h, w, _ = image.shape
    for box, cls, score in zip(boxes, classes, scores):
        if score < threshold:
            continue

        ymin, xmin, ymax, xmax = box
        x1, y1 = int(xmin * w), int(ymin * h)
        x2, y2 = int(xmax * w), int(ymax * h)
        x1, y1 = max(x1, 0),      max(y1, 0)
        x2, y2 = min(x2, w - 1),  min(y2, h - 1)

        label = f"{class_names[cls]}: {score:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            label,
            (x1, max(15, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
    return image

def generate_frames():
    # 모델과 카메라 초기화
    interpreter = Interpreter(model_path='model.tflite')
    interpreter.allocate_tensors()
    class_names = ['fire', 'smoke']

    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate  = 10
    camera.vflip      = True
    raw_capture       = PiRGBArray(camera, size=camera.resolution)
    time.sleep(0.1)  # 카메라 워밍업

    try:
        for frame in camera.capture_continuous(raw_capture, format='bgr', use_video_port=True):
            image_bgr = frame.array
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            boxes, classes, scores = run_inference(interpreter, image_rgb)
            vis = draw_predictions(image_bgr, boxes, classes, scores, class_names)

            ret, jpeg = cv2.imencode('.jpg', vis)
            if not ret:
                raw_capture.truncate(0)
                continue

            frame_bytes = jpeg.tobytes()
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' +
                frame_bytes +
                b'\r\n'
            )
            raw_capture.truncate(0)
    finally:
        camera.close()

@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
