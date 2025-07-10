import streamlit as st
import cv2
import numpy as np
import requests
import time
import threading
import os
from PIL import Image
from tensorflow.lite.python.interpreter import Interpreter as TFLiteInterpreter
from chatbot_helper import DisasterMessageGenerator

# --- Configuration ---
PI_ADDR = "Flask_주소_입력"
PI_STREAM_URL = f"{PI_ADDR}/camera"
PI_TRACK_START_URL = f"{PI_ADDR}/track/start"
PI_TRACK_STOP_URL = f"{PI_ADDR}/track/stop"
PI_DETECT_START_URL = f"{PI_ADDR}/detect/start"
PI_DETECT_STOP_URL = f"{PI_ADDR}/detect/stop"
PI_ARUCO_START_URL = f"{PI_ADDR}/aruco/start"
PI_ARUCO_STOP_URL = f"{PI_ADDR}/aruco/stop"
PI_ARUCO_STATUS_URL = f"{PI_ADDR}/aruco_status"
PI_TEMP_URL = f"{PI_ADDR}/temperature"
PI_HUMID_URL = f"{PI_ADDR}/humidity"
PI_GET_PID_PARAMS_URL = f"{PI_ADDR}/get_pid_params"
PI_SET_PID_PARAMS_URL = f"{PI_ADDR}/set_pid_params"

TARGET_CLASS_ID = 0
DETECTION_THRESHOLD = 0.5
DETECTION_DURATION = 1.2  # seconds

TFLITE_MODEL_PATH = "model1.tflite"
LAST_DETECT_IMAGE_PATH = os.path.join(os.getcwd(), "last_detect.jpg")

# New: Class names mapping
CLASS_NAMES = {
    0: "fire",
    1: "smoke" # Assuming class ID 1 is smoke
}

# --- Thread-Safe Shared State ---
class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.detection_enabled = False
        self.tracking_enabled = False
        self.aruco_enabled = False
        self.is_detecting = False
        self.detection_start_time = None
        self.stop_sent = False
        self.image_ready = False
        self.message_and_image_locked = False
        self.latest_frame_data = np.zeros((480, 640, 3), dtype=np.uint8)
        self.latest_detection_details = None
        self.disaster_message = None
        self.last_detected_image_data = None
        self.aruco_log = []
        self.aruco_tracking = {}
        self.aruco_status = "No logs yet."
        self.latest_aruco_id = None
        self.last_detected_aruco_id = None
        self.disaster_message_generated = False
        self.temperature = None
        self.humidity = None
        # Patrol state
        self.patrol_started = False
        self.patrol_start_marker_id = None
        self.patrolled_markers = []
        self.patrol_events = []
        self.patrol_report = None
        self.prev_is_detecting = False # New: To track previous detection state

    def update_frame(self, frame):
        self.latest_frame_data = frame

    def get_frame(self):
        return self.latest_frame_data

@st.cache_resource
def load_resources():
    try:
        interpreter = TFLiteInterpreter(model_path=TFLITE_MODEL_PATH)
        interpreter.allocate_tensors()
        chatbot = DisasterMessageGenerator()
        return interpreter, chatbot
    except Exception as e:
        st.error(f"Failed to load resources: {e}")
        st.stop()

interpreter, chatbot = load_resources()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_h, input_w = input_details[0]['shape'][1:3]

if 'shared_state' not in st.session_state:
    st.session_state.shared_state = SharedState()
if 'app_started' not in st.session_state:
    st.session_state.app_started = False
if 'aruco_thread_started' not in st.session_state:
    st.session_state.aruco_thread_started = False

shared_state = st.session_state.shared_state

# --- Core Functions ---
def send_pi_command(url):
    try:
        requests.post(url, timeout=2)
    except requests.exceptions.RequestException:
        pass

# PID 파라미터 가져오기
def get_pid_params():
    try:
        response = requests.get(PI_GET_PID_PARAMS_URL, timeout=2)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to get PID parameters: {response.status_code}")
            return None
    except requests.exceptions.RequestException:
        st.error("Could not connect to the PiCar-4WD server to get PID params.")
        return None

# PID 파라미터 업데이트
def set_pid_params(params):
    try:
        response = requests.post(PI_SET_PID_PARAMS_URL, json=params, timeout=2)
        if response.status_code == 200:
            pass # Removed st.success message
        else:
            st.error(f"Failed to set PID parameters: {response.status_code}, Response: {response.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"Could not connect to the PiCar-4WD server to set PID params: {e}")

def mjpeg_stream():
    try:
        stream = requests.get(PI_STREAM_URL, stream=True, timeout=10)
        buf = b""
        for chunk in stream.iter_content(chunk_size=1024):
            buf += chunk
            a = buf.find(b'\xff\xd8')
            b = buf.find(b'\xff\xd9')
            if a != -1 and b != -1:
                jpg = buf[a:b+2]
                buf = buf[b+2:]
                frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                yield frame
    except:
        while True:
            yield None

def detect(frame):
    img = cv2.resize(frame, (input_w, input_h))
    interpreter.set_tensor(input_details[0]['index'], np.expand_dims(img,0).astype(np.uint8))
    interpreter.invoke()
    scores = interpreter.get_tensor(output_details[0]['index'])[0]
    boxes = interpreter.get_tensor(output_details[1]['index'])[0]
    num_raw = interpreter.get_tensor(output_details[2]['index'])
    classes = interpreter.get_tensor(output_details[3]['index'])[0].astype(np.int32)
    n = int(num_raw[0]) if num_raw.size>0 else boxes.shape[0]
    return boxes[:n], classes[:n], scores[:n]

def monitor_detection_thread(state):
    while True:
        to_process = None
        with state.lock:
            if state.detection_enabled and state.is_detecting:
                if state.detection_start_time is None:
                    state.detection_start_time = time.time()
                elif time.time() - state.detection_start_time > DETECTION_DURATION:
                    if not state.stop_sent and not state.message_and_image_locked and not state.disaster_message_generated:
                            send_pi_command(PI_TRACK_STOP_URL)
                            state.stop_sent = True
                            to_process = state.latest_detection_details
                            frame = state.get_frame()
                            if frame is not None:
                                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                state.last_detected_image_data = img
                                state.image_ready = True
                                state.message_and_image_locked = True
            else:
                state.detection_start_time = None
                state.stop_sent = False
                state.message_and_image_locked = False
                state.latest_detection_details = None

        if to_process and not state.disaster_message_generated:
            current_time_str = time.strftime("%H:%M:%S")
            aruco_id_for_message = "알 수 없음"
            with state.lock:
                if state.last_detected_aruco_id is not None:
                    aruco_id_for_message = str(state.last_detected_aruco_id)

            msg = chatbot.generate_message(
                time=current_time_str,
                aruco_id=aruco_id_for_message,
                box_width=to_process['box'][2] - to_process['box'][0],
                box_height=to_process['box'][3] - to_process['box'][1],
                score=to_process['score'],
                detected_object_type=to_process['class_name']
            )
            with state.lock:
                state.disaster_message = msg
                state.disaster_message_generated = True
                detection_type_korean = "화재" if to_process['class_name'] == "fire" else "연기"
                event_desc = f"{current_time_str}: 구간 {aruco_id_for_message}에서 {detection_type_korean} 감지 (예측 확률: {to_process['score']:.2f})"
                state.patrol_events.append(event_desc)
                state.aruco_log.append(event_desc) # Add to aruco_log

        time.sleep(0.5)

def video_processing_thread(state):
    for frame in mjpeg_stream():
        if frame is None:
            time.sleep(0.1)
            continue
        detect_flag = False
        with state.lock:
            de = state.detection_enabled
        if de:
            boxes, classes, scores = detect(frame)
            h,w = frame.shape[:2]
            for i in range(len(scores)):
                if scores[i]>=DETECTION_THRESHOLD:
                    y1,x1,y2,x2 = boxes[i]
                    l,t,r,b = int(x1*w),int(y1*h),int(x2*w),int(y2*h)
                    class_id = classes[i]
                    class_name = CLASS_NAMES.get(class_id, "Unknown")
                    label = f"{class_name}: {scores[i]:.2f}"
                    color = (0, 0, 255) if class_name == "fire" else (0, 255, 255) # Red for fire, Yellow for smoke
                    cv2.rectangle(frame,(l,t),(r,b),color,2)
                    cv2.putText(frame,label,(l,t-5),cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)
                    
                    # Only consider fire or smoke for detection logic
                    if class_name == "fire" or class_name == "smoke":
                        detect_flag = True
                        with state.lock:
                            state.latest_detection_details = {'score':scores[i],'box':[l,t,r,b], 'class_name': class_name}
                        break
            with state.lock:
                state.is_detecting = detect_flag
        state.update_frame(frame)

def aruco_status_loop(state):
    while True:
        if not state.aruco_enabled:
            time.sleep(0.1)
            continue

        try:
            res = requests.get(PI_ARUCO_STATUS_URL, timeout=1).json()
            markers = res.get('aruco_markers', [])
            if not markers:
                time.sleep(0.1)
                continue

            marker_id = markers[0]['id']
            current_time_str = time.strftime("%H:%M:%S")

            with state.lock:
                state.last_detected_aruco_id = marker_id

                # 순찰 시작
                if not state.patrol_started:
                    state.patrol_started = True
                    state.patrol_start_marker_id = marker_id
                    state.patrolled_markers = [(current_time_str, marker_id)]
                    state.patrol_events = []
                    state.patrol_report = None
                    log_entry = f"{current_time_str}: 순찰 시작 (시작점: {marker_id})"
                    state.aruco_log.append(log_entry)

                else:
                    # 마지막에 인식된 마커와 다른 마커를 새로 인식한 경우
                    last_marker_id = state.patrolled_markers[-1][1]
                    if marker_id != last_marker_id:
                        # 시작 마커로 복귀했고 경로가 두 개 이상인 경우 → 순찰 종료
                        if marker_id == state.patrol_start_marker_id and len(state.patrolled_markers) > 1:
                            state.patrolled_markers.append((current_time_str, marker_id))
                            path_str = ' -> '.join([str(m_id) for t, m_id in state.patrolled_markers])
                            log_entry = f"{current_time_str}: 순찰 완료. 시스템을 정지합니다."
                            state.aruco_log.append(log_entry)

                            # 보고서 생성
                            report = chatbot.generate_patrol_report(
                                patrolled_markers=state.patrolled_markers,
                                patrol_events=state.patrol_events
                            )
                            state.patrol_report = report

                            # 시스템 정지
                            send_pi_command(PI_TRACK_STOP_URL)
                            send_pi_command(PI_ARUCO_STOP_URL)
                            send_pi_command(PI_DETECT_STOP_URL)

                            state.tracking_enabled = False
                            state.aruco_enabled = False
                            state.detection_enabled = False

                            # 상태 초기화
                            state.patrol_started = False
                            state.patrol_start_marker_id = None
                            state.patrolled_markers = []
                            state.patrol_events = []
                        else:
                            # 새로운 마커를 경로에 추가
                            state.patrolled_markers.append((current_time_str, marker_id))
                            log_entry = f"{current_time_str}: {marker_id} 구간 통과"
                            state.aruco_log.append(log_entry)

        except requests.exceptions.RequestException as e:
            print(f"[ARUCO THREAD] Failed to get ArUco data: {e}")
            time.sleep(1)
            continue
        time.sleep(0.1)


def sensor_data_loop(state):
    while True:
        try:
            temp_res = requests.get(PI_TEMP_URL, timeout=1).json()
            humid_res = requests.get(PI_HUMID_URL, timeout=1).json()
            with state.lock:
                state.temperature = temp_res.get("temperature")
                state.humidity = humid_res.get("humidity")
        except requests.exceptions.RequestException:
            with state.lock:
                state.temperature = None
                state.humidity = None
        time.sleep(5)

def main():
    st.set_page_config(page_title="Fire Detection Control", layout="wide")
    st.title("🔥 Fire Detection Control Panel")

    if not st.session_state.app_started:
        threading.Thread(target=monitor_detection_thread, args=(shared_state,), daemon=True).start()
        threading.Thread(target=video_processing_thread, args=(shared_state,), daemon=True).start()
        st.session_state.app_started = True

    if not st.session_state.aruco_thread_started:
        threading.Thread(target=aruco_status_loop, args=(shared_state,), daemon=True).start()
        st.session_state.aruco_thread_started = True

    if 'sensor_thread_started' not in st.session_state:
        st.session_state.sensor_thread_started = False

    if not st.session_state.sensor_thread_started:
        threading.Thread(target=sensor_data_loop, args=(shared_state,), daemon=True).start()
        st.session_state.sensor_thread_started = True

    with st.sidebar:
        st.header("⚙️ Controls")
        if st.button("Tracking Start", use_container_width=True):
            send_pi_command(PI_TRACK_START_URL)
            shared_state.tracking_enabled = True
        if st.button("Tracking Stop", use_container_width=True):
            send_pi_command(PI_TRACK_STOP_URL)
            shared_state.tracking_enabled = False
        st.divider()
        if st.button("Fire Detection Start", use_container_width=True):
            send_pi_command(PI_DETECT_START_URL)
            with shared_state.lock:
                shared_state.detection_enabled = True
                shared_state.image_ready = False
                shared_state.disaster_message = None
                shared_state.message_and_image_locked = False
                shared_state.last_detected_image_data = None
                shared_state.disaster_message_generated = False

        if st.button("Fire Detection Stop", use_container_width=True):
            send_pi_command(PI_DETECT_STOP_URL)
            with shared_state.lock:
                shared_state.detection_enabled = False
                shared_state.disaster_message_generated = False

        st.divider()
        st.header("📍 ArUco Marker")
        if st.button("ArUco Detection Start", use_container_width=True):
            send_pi_command(PI_ARUCO_START_URL)
            with shared_state.lock:
                shared_state.aruco_enabled = True
                shared_state.patrol_report = None
                shared_state.aruco_log = []
                shared_state.aruco_status = "ArUco detection is ON. Waiting for markers..."

        if st.button("ArUco Detection Stop", use_container_width=True):
            send_pi_command(PI_ARUCO_STOP_URL)
            with shared_state.lock:
                shared_state.aruco_enabled = False
                shared_state.patrol_started = False
                shared_state.patrol_start_marker_id = None
                shared_state.patrolled_markers = []
                shared_state.patrol_events = []
                shared_state.aruco_status = "ArUco detection is OFF."

        st.divider()
        st.write("Detected Log:")
        aruco_status_placeholder = st.empty()

        st.divider()
        initial_params = get_pid_params()
        if initial_params:
            st.session_state.TRACK_LINE_SPEED = initial_params.get("TRACK_LINE_SPEED", 25)
        else:
            st.session_state.TRACK_LINE_SPEED = 25

        st.subheader("Base Speed")
        new_speed = st.number_input(
            "Enter Base Speed",
            min_value=0,
            max_value=50,
            value=int(st.session_state.TRACK_LINE_SPEED),
            step=1,
            key="speed_input"
        )

        if new_speed != st.session_state.TRACK_LINE_SPEED:
            st.success(f"Base Speed changed from {st.session_state.TRACK_LINE_SPEED} to {new_speed}")
            st.session_state.TRACK_LINE_SPEED = new_speed
            set_pid_params({"TRACK_LINE_SPEED": new_speed})

        st.divider()
        st.header("Current Status")
        status_placeholder = st.empty()

    col1, col2 = st.columns(2)
    with col1:
        st.header("Live Feed")
        live = st.empty()
    with col2:
        st.header("Last Detected Image")
        last = st.empty()

    msg_cont = st.empty()
    report_cont = st.empty()
    latest_frame = np.zeros((480,640,3),dtype=np.uint8)

    # st.header("💬Chatbot")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("궁금한 점을 질문하세요:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("답변 생성 중..."):
            current_status = {
                "temperature": shared_state.temperature,
                "humidity": shared_state.humidity,
                "tracking_enabled": shared_state.tracking_enabled,
                "detection_enabled": shared_state.detection_enabled,
                "aruco_enabled": shared_state.aruco_enabled,
                "last_detected_aruco_id": shared_state.last_detected_aruco_id,
                "latest_detection_details": shared_state.latest_detection_details,
                "latest_disaster_message": shared_state.disaster_message
            }
            response = chatbot.answer_question(prompt, current_status)
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)

    while True:
        frame = shared_state.get_frame()
        if frame is not None:
            latest_frame = frame
        rgb = cv2.cvtColor(latest_frame,cv2.COLOR_BGR2RGB)
        live.image(rgb, use_container_width=True)
        if shared_state.image_ready and shared_state.last_detected_image_data is not None:
            last.image(shared_state.last_detected_image_data, use_container_width=True)
        else:
            last.info("No detection image.")
        
        with shared_state.lock:
            dm = shared_state.disaster_message
            pr = shared_state.patrol_report

            # Centralized update of aruco_status from aruco_log
            if shared_state.aruco_log:
                shared_state.aruco_status = "\n".join(shared_state.aruco_log)
            elif shared_state.aruco_enabled:
                shared_state.aruco_status = "ArUco detection is ON. Waiting for markers..."
            else:
                shared_state.aruco_status = "ArUco detection is OFF."

            status_text = shared_state.aruco_status

        if dm:
            msg_cont.error(dm)
        if pr:
            report_cont.success(pr)
        
        aruco_status_placeholder.code(status_text)

        # Reset disaster_message_generated when detection stops
        current_is_detecting = shared_state.is_detecting
        if shared_state.prev_is_detecting and not current_is_detecting:
            with shared_state.lock:
                shared_state.disaster_message_generated = False

        shared_state.prev_is_detecting = current_is_detecting

        current_speed_val = st.session_state.TRACK_LINE_SPEED
        tracking_status = "ON" if shared_state.tracking_enabled else "OFF"
        detection_status = "ON" if shared_state.detection_enabled else "OFF"
        aruco_status_display = "ON" if shared_state.aruco_enabled else "OFF"

        status_placeholder.markdown(f"""
        - **Line Tracking:** {tracking_status}
        - **Detection:** {detection_status}
        - **ArUco:** {aruco_status_display}
        - **Speed:** {current_speed_val}
        """)

        time.sleep(0.01)

if __name__=="__main__":
    main()
