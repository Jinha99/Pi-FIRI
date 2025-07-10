#!/usr/bin/env python3
"""
Simple camera stream test using Flask and PiCamera.
This example sets up a web server that streams video from the Raspberry Pi camera.
"""
from flask import Flask, Response
from picamera import PiCamera
import io

app = Flask(__name__)

# Initialize camera with desired resolution
camera = PiCamera()
camera.vflip = True  # 상하 반전
camera.resolution = (640, 480)

@app.route('/')
def index():
    return '''
        <h2>Camera Streaming Server</h2>
        <p>영상 스트리밍을 보려면 <a href="/camera">/camera</a>로 접속하세요.</p>
    '''

def generate_frames():
    """Video streaming generator function."""
    stream = io.BytesIO()
    for _ in camera.capture_continuous(stream, format='jpeg', use_video_port=True):
        stream.seek(0)
        frame = stream.read()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        stream.seek(0)
        stream.truncate()


@app.route('/camera')
def camera_feed():
    """Return the camera stream as an MJPEG feed."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    # Run the server on all interfaces so other machines can access it
    app.run(host='0.0.0.0', port=8000, threaded=True)
