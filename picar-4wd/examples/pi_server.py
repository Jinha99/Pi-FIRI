"""
Server from raspberry pi to control a car with line tracking, ArUco marker detection, and sensor readings.
This server uses Flask to provide a web interface for controlling the car and retrieving sensor data.
It includes endpoints for starting/stopping tracking, ArUco detection, and retrieving camera feed.
"""

#!/usr/bin/env python3
from threading import Thread, Lock
import time
import io
import signal
import sys
from flask import Flask, Response, request, jsonify
from picamera import PiCamera
import picar_4wd as fc
import cv2
import numpy as np
import Adafruit_DHT

app = Flask(__name__)

# 상태 변수들
running = True
tracking_enabled = False
detecting_enabled = False
aruco_enabled = False
latest_frame = None
aruco_data = []
frame_lock = Lock()
sensor_lock = Lock()

# 설정값 (전역 변수로 선언)
TRACK_LINE_SPEED = 25

# PID 상수 (전역 변수로 선언)
Kp_st = 0.45
Ki_st = 0.03
Kd_st = 0.15
Kp_arc = 0.2
Ki_arc = 0.01
Kd_arc = 0.07

@app.route('/get_pid_params', methods=['GET'])
def get_pid_params():
    global TRACK_LINE_SPEED, Kp_st, Ki_st, Kd_st, Kp_arc, Ki_arc, Kd_arc
    return jsonify({
        "TRACK_LINE_SPEED": TRACK_LINE_SPEED,
        "Kp_st": Kp_st, "Ki_st": Ki_st, "Kd_st": Kd_st,
        "Kp_arc": Kp_arc, "Ki_arc": Ki_arc, "Kd_arc": Kd_arc
    }), 200

@app.route('/set_pid_params', methods=['POST'])
def set_pid_params():
    global TRACK_LINE_SPEED, Kp_st, Ki_st, Kd_st, Kp_arc, Ki_arc, Kd_arc
    data = request.json
    if "TRACK_LINE_SPEED" in data:
        TRACK_LINE_SPEED = float(data["TRACK_LINE_SPEED"])
    if "Kp_st" in data:
        Kp_st = float(data["Kp_st"])
    if "Ki_st" in data:
        Ki_st = float(data["Ki_st"])
    if "Kd_st" in data:
        Kd_st = float(data["Kd_st"])
    if "Kp_arc" in data:
        Kp_arc = float(data["Kp_arc"])
    if "Ki_arc" in data:
        Ki_arc = float(data["Ki_arc"])
    if "Kd_arc" in data:
        Kd_arc = float(data["Kd_arc"])
    return jsonify({"status": "success", "message": "PID parameters updated"}), 200

@app.route('/set_tracking_status', methods=['POST'])
def set_tracking_status():
    global tracking_enabled
    data = request.json
    if "enabled" in data:
        tracking_enabled = bool(data["enabled"])
        if not tracking_enabled:
            fc.stop()
        return jsonify({"status": "success", "tracking_enabled": tracking_enabled}), 200
    return jsonify({"status": "error", "message": "Invalid request"}), 400


# 카메라 설정
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 7
camera.vflip = True
camera.hflip = True

# ArUco 설정 (최신 OpenCV 호환 코드)
try:
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    aruco_params = cv2.aruco.DetectorParameters()
    aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    use_new_aruco = True
except AttributeError:
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
    aruco_params = cv2.aruco.DetectorParameters_create()
    use_new_aruco = False

@app.route('/track/start', methods=['POST'])
def start_tracking():
    global tracking_enabled
    tracking_enabled = True
    return '', 204

@app.route('/track/stop', methods=['POST'])
def stop_tracking():
    global tracking_enabled
    tracking_enabled = False
    fc.stop()
    return '', 204

@app.route('/aruco/start', methods=['POST'])
def start_aruco():
    global aruco_enabled
    aruco_enabled = True
    return '', 204

@app.route('/aruco/stop', methods=['POST'])
def stop_aruco():
    global aruco_enabled
    aruco_enabled = False
    return '', 204

@app.route('/aruco_status', methods=['GET'])
def aruco_status():
    global aruco_data
    return jsonify({"aruco_markers": aruco_data}), 200

@app.route('/temperature', methods=['GET'])
def get_temperature():
    global temperature
    with sensor_lock:
        return jsonify({"temperature": temperature}), 200

@app.route('/humidity', methods=['GET'])
def get_humidity():
    global humidity
    with sensor_lock:
        return jsonify({"humidity": humidity}), 200

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

def camera_loop():
    global latest_frame, aruco_data
    stream = io.BytesIO()
    for _ in camera.capture_continuous(stream, format='jpeg', use_video_port=True, quality=80):
        if not running:
            break
        stream.seek(0)
        jpg_data = stream.read()

        frame = cv2.imdecode(np.frombuffer(jpg_data, np.uint8), cv2.IMREAD_COLOR)

        if aruco_enabled:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if use_new_aruco:
                corners, ids, _ = aruco_detector.detectMarkers(gray)
            else:
                corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

            if ids is not None:
                # 가장 큰 마커 찾기
                max_area = 0
                best_id = -1
                best_corners = None
                
                for i, corner_set in enumerate(corners):
                    # 각 마커의 면적 계산
                    area = cv2.contourArea(corner_set)
                    if area > max_area:
                        max_area = area
                        best_id = ids[i][0]
                        best_corners = corner_set
                
                if best_id != -1:
                    # 가장 큰 마커만 그리기 및 데이터 저장
                    cv2.aruco.drawDetectedMarkers(frame, [best_corners], np.array([[best_id]]))
                    # Exclude aruco_id 190 and 158 from aruco_data
                    if best_id not in [190, 158]:
                        aruco_data = [{"id": int(best_id), "corners": best_corners.tolist()}]
                    else:
                        aruco_data = []
                else:
                    aruco_data = []
            else:
                aruco_data = []

        # 마커가 그려진 프레임을 다시 JPEG으로 인코딩하여 latest_frame에 저장
        _, encoded_frame = cv2.imencode('.jpg', frame)
        with frame_lock:
            latest_frame = encoded_frame.tobytes()

        stream.seek(0)
        stream.truncate()


def track_line_loop():
    global Kp_st, Ki_st, Kd_st, Kp_arc, Ki_arc, Kd_arc, TRACK_LINE_SPEED

    
    threshold_pivot = 100 # 회전 임계값
    pivot_power = 40 # 회전 시 모터 파워

    arc_radius = 200 # 호 회전 반경
    threshold_arc = 100 # 호 회전 임계값

    delta = 30 # 호 회전 임계값과 일반 회전 임계값 사이의 차이

    prev_error = 0
    integral = 0

    # 라인 미검출 관련 변수
    last_line_time = time.time()
    line_missing = False
    LINE_MISS_THRESHOLD_SEC = 0.5
    BACKWARD_TIME = 0.7           # 후진 시간(초)
    BACKWARD_POWER = TRACK_LINE_SPEED            # 후진 속도

    while running:
        if not tracking_enabled:
            fc.stop()
            time.sleep(0.02)
            continue

        gs = fc.get_grayscale_list()
        error = gs[0] - gs[2]

        # 라인 인식 실패 판정 (센서 3개가 모두 1300 이상이면 라인 미감지로 간주)
        if gs[0] > 1300 and gs[1] > 1300 and gs[2] > 1300:
            if not line_missing:
                last_line_time = time.time()
                line_missing = True
            elif time.time() - last_line_time > LINE_MISS_THRESHOLD_SEC:
                fc.backward(BACKWARD_POWER)
                time.sleep(BACKWARD_TIME)
                fc.stop()
                last_line_time = time.time()
                continue
        else:
            line_missing = False

        if abs(error) > threshold_pivot:
            if error > 0:
                l, r = -pivot_power, pivot_power
            else:
                l, r = pivot_power, -pivot_power
        elif threshold_arc < abs(error) <= threshold_arc + delta:
            ratio = (abs(error) - threshold_arc) / delta
            current_speed = TRACK_LINE_SPEED * (1 - 0.3 * ratio)
            turn_rate = (error / arc_radius) * ratio 
            l = current_speed - turn_rate * current_speed
            r = current_speed + turn_rate * current_speed
            integral += error
            integral = max(-500, min(500, integral))
            derivative = error - prev_error
            corr = (Kp_arc*error + Ki_arc*integral + Kd_arc*derivative) * ratio
            prev_error = error
            l = max(0, min(100, l - corr))
            r = max(0, min(100, r + corr))
        elif abs(error) > threshold_arc + delta:
            turn_rate = error / arc_radius 
            current_speed = TRACK_LINE_SPEED * 0.7
            l = current_speed - turn_rate * current_speed
            r = current_speed + turn_rate * current_speed
            integral += error
            integral = max(-500, min(500, integral))
            derivative = error - prev_error
            corr = (Kp_arc*error + Ki_arc*integral + Kd_arc*derivative)
            prev_error = error
            l = max(0, min(100, l - corr))
            r = max(0, min(100, r + corr))
        else:
            integral += error
            integral = max(-500, min(500, integral))
            derivative = error - prev_error
            corr = (Kp_st*error + Ki_st*integral + Kd_st*derivative)
            prev_error = error
            l = max(0, min(100, TRACK_LINE_SPEED - corr))
            r = max(0, min(100, TRACK_LINE_SPEED + corr))

        fc.set_motor_power(2, l)
        fc.set_motor_power(4, l)
        fc.set_motor_power(1, r)
        fc.set_motor_power(3, r)
        time.sleep(0.02)


def sensor_reading_loop():
    global temperature, humidity
    DHT_SENSOR = Adafruit_DHT.DHT11
    DHT_PIN = 17 # DHT 센서가 연결된 GPIO 핀 번호

    while running:
        h, t = Adafruit_DHT.read_retry(DHT_SENSOR, DHT_PIN)
        if t is not None and h is not None:
            with sensor_lock:
                temperature = t
                humidity = h
        else:
            print("[Temp Sensor] DHT11 reading failed")
        time.sleep(5)

def signal_handler(sig, frame):
    global running
    running = False
    fc.stop()
    sys.exit(0)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    fc.start_speed_thread()

    Thread(target=camera_loop, daemon=True).start()
    Thread(target=track_line_loop, daemon=True).start()
    Thread(target=sensor_reading_loop, daemon=True).start()

    app.run(host='0.0.0.0', port=8000, threaded=True)
