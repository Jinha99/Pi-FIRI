# Pi-FIRI: 라즈베리파이 기반 화재 감지 지능형 로봇

## 프로젝트 개요

**Pi-FIRI**는 Raspberry Pi 4와 Picar-4WD 로봇 키트를 기반으로, 실시간 영상 스트리밍과 경량 딥러닝 모델(TFLite EfficientDet-Lite1)을 이용해 화재·연기를 탐지하고, 감지 시 자연어 경고 메시지를 생성·전송하는 지능형 감시 시스템입니다.

## 핵심 기능

- **실시간 화재·연기 감지**
  - 라즈베리파이 카메라로 전송되는 MJPEG 스트림을 프레임 단위로 수신

  - 수신된 프레임을 TFLite 모델(EfficientDet-Lite)로 분석하여 화재 또는 연기 객체 탐지

  - 동일 객체가 일정 시간 이상 연속으로 감지되면 경고 알림을 트리거

- **자율 주행 및 순찰**
  - 바닥 라인을 따라 주행하는 PID 기반 라인트레이싱 제어

  - 주행 중 ArUco 마커를 인식하여 순찰 경로 상의 위치를 파악하고 기록

  - API를 통해 주행 시작/정지 및 순찰 모드를 유연하게 전환 가능

- **자연어 경고 및 순찰 보고서 생성**
  - 감지된 시간, 위치, 객체 정보 등을 기반으로 LLM(Upstage API)에 상황 전달

  - 화재 감지 시 자연어 경고 메시지 생성

  - 순찰 완료 후 감지 요약과 권고 사항이 포함된 보고서 자동 생성

- **Streamlit 기반 대시보드**
  - 실시간 영상 스트리밍, 마지막 탐지 이미지, 온·습도 센서 데이터 표시

  - PID 속도 조절, 탐지/주행 모드 전환 등의 제어 UI 제공

  - 사용자의 질의에 응답하는 채팅 인터페이스 내장


## 디렉토리 구조
```
.
├── local/
│ ├── local_server.py # Streamlit UI + 제어 로직
│ ├── chatbot_helper.py # 화재 감지 메시지·순찰 보고서 생성
│ └── model1.tflite # EfficientDet-Lite1 화재 감지 모델
├── picar-4wd/ # Picar-4WD 제어 라이브러리
│ └── picar_4wd/
└── pi_server.py # Raspberry Pi Flask 서버
```

## 주요 파일 설명

| 파일                              | 역할                                                         |
|-----------------------------------|--------------------------------------------------------------|
| `local/local_server.py`           | Streamlit 기반 대시보드 앱 (영상 처리, 탐지, 센서·모드 제어)     |
| `local/chatbot_helper.py`         | Upstage LLM 연동: 감지 알림·순찰 보고서 생성                    |
| `local/model0.tflite`             | EfficientDet-Lite1 화재·연기 탐지 TFLite 모델                  |
| `local/last_detect.jpg`           | 마지막 감지 시 프레임 저장                                   |
| `pi_server.py`                    | Flask 서버: 카메라 스트림, PID 주행, ArUco·센서 API 제공        |
| `picar-4wd/picar_4wd/...`         | 모터·PWM 제어, 센서 인터페이스 등 로봇 제어 기능                |


## 설치 및 실행

1. **Raspberry Pi 서버 실행**  
    ```bash
    git clone <repo-url>
    cd <repo-root>
    pip3 install -r requirements.txt
    python3 pi_server.py    # 포트 8000
    ```

2. **로컬 PC 대시보드 실행**  
    ```bash
    cd <repo-root>/local
    pip install -r requirements.txt
    streamlit run local_server.py  # 기본 포트 8501
    ```

3. **웹 브라우저 접속**  
    ```
    http://localhost:8501
    ```

4. **Streamlit 좌측 사이드바**  
    - Tracking Start / Stop  
    - Fire Detection Start / Stop  
    - ArUco Detection Start / Stop  
    - Base Speed 조절  

## 실행 예시
