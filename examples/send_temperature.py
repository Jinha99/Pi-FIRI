#!/usr/bin/env python3
from threading import Thread, Lock
import time
import io
import signal
import sys
from flask import Flask, Response, request, jsonify
from picamera import PiCamera
import picar_4wd as fc
import Adafruit_DHT

app = Flask(__name__)

# 상태 변수들
running = True
tracking_enabled = False
detecting_enabled = False
overheat = False
current_temp = 0.0
latest_frame = None
frame_lock = Lock()

# 설정값
TEMP_THRESHOLD = 20.0  # °C
TRACK_LINE_SPEED = 30

# 카메라 설정
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 10
camera.vflip = True

@app.route('/track/start', methods=['POST'])
def start_tracking():
    global tracking_enabled
    if not overheat:
        tracking_enabled = True
        print("[INFO] Tracking START")
    return '', 204

@app.route('/track/stop', methods=['POST'])
def stop_tracking():
    global tracking_enabled
    tracking_enabled = False
    fc.stop()
    print("[INFO] Tracking STOP")
    return '', 204

@app.route('/detect/start', methods=['POST'])
def start_detection():
    global detecting_enabled
    detecting_enabled = True
    print("[INFO] Detection ENABLED")
    return '', 204

@app.route('/detect/stop', methods=['POST'])
def stop_detection():
    global detecting_enabled
    detecting_enabled = False
    print("[INFO] Detection DISABLED")
    return '', 204

@app.route('/camera')
def camera_feed():
    def stream():
        while running:
            with frame_lock:
                if latest_frame:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + latest_frame + b'\r\n')
            time.sleep(0.1)
    return Response(stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/temperature')
def get_temperature():
    return jsonify({
        'temperature': current_temp,
        'overheat': overheat
    })

def camera_loop():
    global latest_frame
    stream = io.BytesIO()
    frame_counter = 0

    for _ in camera.capture_continuous(stream, format='jpeg', use_video_port=True, quality=85):
        if not running:
            break
        stream.seek(0)
        jpg_data = stream.read()
        with frame_lock:
            latest_frame = jpg_data
        if detecting_enabled and frame_counter % 5 == 0:
            print("[INFO] Detection frame triggered")
        stream.seek(0)
        stream.truncate()
        frame_counter += 1

def track_line_loop():
    while running:
        if tracking_enabled:
            try:
                gs_list = fc.get_grayscale_list()
                status = fc.get_line_status(1200, gs_list)
                if status == 0:
                    fc.forward(TRACK_LINE_SPEED)
                elif status == -1:
                    fc.turn_left(TRACK_LINE_SPEED)
                elif status == 1:
                    fc.turn_right(TRACK_LINE_SPEED)
                else:
                    fc.forward(TRACK_LINE_SPEED)
            except OSError as e:
                print(f"[Line Sensor] error: {e}")
        else:
            fc.stop()
        time.sleep(0.02)

def temperature_loop():
    global current_temp, overheat, tracking_enabled
    DHT_SENSOR = Adafruit_DHT.DHT11
    DHT_PIN = 17

    while running:
        humidity, temp_c = Adafruit_DHT.read_retry(DHT_SENSOR, DHT_PIN)
        if temp_c is not None:
            current_temp = temp_c
            if current_temp >= TEMP_THRESHOLD:
                overheat = True
                tracking_enabled = False
                fc.stop()
                print(f"[TEMP] Overheat detected: {current_temp} °C")
            else:
                overheat = False
        else:
            print("[Temp Sensor] DHT11 reading failed")
        time.sleep(1)

def signal_handler(sig, frame):
    global running
    print("SIGINT received. Exiting...")
    running = False
    fc.stop()
    sys.exit(0)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    fc.start_speed_thread()

    Thread(target=camera_loop, daemon=True).start()
    Thread(target=track_line_loop, daemon=True).start()
    # Thread(target=temperature_loop, daemon=True).start() 온도 측정 일단 보류

    app.run(host='0.0.0.0', port=8000, threaded=True)
