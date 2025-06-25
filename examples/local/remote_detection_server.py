import cv2
import numpy as np
import requests
import time
import threading
from tensorflow.lite.python.interpreter import Interpreter as TFLiteInterpreter

# MJPEG 스트림 URL
PI_STREAM_URL = "http://192.168.138.115:8000/camera"
PI_STOP_URL   = "http://192.168.138.115:8000/track/stop"

# 감지 조건
TARGET_CLASS_ID    = 0    # 예: person
DETECTION_THRESHOLD= 0.5
DETECTION_DURATION = 1.5  # 초

# TFLite 모델 로드
interpreter = TFLiteInterpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_h, input_w= input_details[0]['shape'][1:3]

last_detection_time = 0
stop_sent = False

def mjpeg_stream():
    stream = requests.get(PI_STREAM_URL, stream=True)
    buf = b""
    for chunk in stream.iter_content(chunk_size=1024):
        buf += chunk
        a = buf.find(b'\xff\xd8')
        b = buf.find(b'\xff\xd9')
        if a!=-1 and b!=-1:
            jpg = buf[a:b+2]
            buf = buf[b+2:]
            frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            yield frame

def detect(frame):
    """TFLite EfficientDet 출력 텐서를 고정 인덱스로 매핑"""
    # 전처리
    img = cv2.resize(frame, (input_w, input_h))
    input_tensor = np.expand_dims(img, 0).astype(np.uint8)
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()

    # 고정 인덱스 매핑
    scores_all  = interpreter.get_tensor(output_details[0]['index'])[0]      # [25]
    boxes_all   = interpreter.get_tensor(output_details[1]['index'])[0]      # [25,4]
    num_raw     = interpreter.get_tensor(output_details[2]['index'])        # (1,)
    classes_all = interpreter.get_tensor(output_details[3]['index'])[0]      # [25]

    # num_detections 읽기
    num_det = int(num_raw[0]) if num_raw.size>0 else boxes_all.shape[0]
    num_det = min(num_det, boxes_all.shape[0])

    # 슬라이싱
    boxes   = boxes_all[:num_det]
    scores  = scores_all[:num_det]
    classes = classes_all[:num_det].astype(np.int32)

    return boxes, classes, scores


detection_start_time = None
stop_sent = False
is_detecting = False

def monitor_detection():
    global detection_start_time, stop_sent, is_detecting
    while True:
        if is_detecting:
            if detection_start_time is None:
                detection_start_time = time.time()
            elif time.time() - detection_start_time > DETECTION_DURATION and not stop_sent:
                print("[INFO] 감지 1.5초 지속 — Pi에 정지 요청")
                requests.post(PI_STOP_URL)  # 실제 정지 요청 보내기
                stop_sent = True
        else:
            detection_start_time = None
            stop_sent = False

        time.sleep(0.1)

if __name__ == "__main__":
    threading.Thread(target=monitor_detection, daemon=True).start()

    for frame in mjpeg_stream():
        boxes, classes, scores = detect(frame)
        detected = False
        h, w = frame.shape[:2]

        for i in range(len(scores)):
            if scores[i] >= DETECTION_THRESHOLD and classes[i] == TARGET_CLASS_ID:
                ymin, xmin, ymax, xmax = boxes[i]
                l, t = int(xmin * w), int(ymin * h)
                r, b = int(xmax * w), int(ymax * h)
                cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
                cv2.putText(frame, f"{classes[i]}:{scores[i]:.2f}", (l, t - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                detected = True

        is_detecting = detected  # <-- 감지 상태 공유

        cv2.imshow("Detection", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()