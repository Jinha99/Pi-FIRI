# Pi-FIRI: Pi-based Fire Robotic Intelligence

## 📌 프로젝트 개요

**Pi-FIRI**는 Raspberry Pi 4와 Picar-4WD 로봇 키트를 기반으로, 실시간 영상 스트리밍과 경량 딥러닝 모델(TFLite EfficientDet-Lite1)을 이용해 화재·연기를 탐지하고, 감지 시 자연어 경고 메시지를 생성·전송하는 지능형 감시 시스템입니다.

**프로젝트 진행 기간**

2025/05/04 ~ 2025/07/11

**팀원 구성**

| 김상우 | 송진하 | 임규원 | 정서영 |
|:---:|:---:|:---:|:---:|
| <img src="https://github.com/Underove.png" width="100"/> | <img src="https://github.com/Jinha99.png" width="100"/> | <img src="https://github.com/gwlim3012.png" width="100"/> | <img src="https://github.com/jnalgae.png" width="100"/> |
| [@Underove](https://github.com/Underove) | [@Jinha99](https://github.com/Jinha99) | [@gwlim3012](https://github.com/gwlim3012) |[@jnalgae](https://github.com/jnalgae) |

**전체 파이프라인**
![image](https://github.com/user-attachments/assets/5c2d4fac-9bd7-4da8-b64f-5607d37a35cf)

## 🔧 사용 하드웨어
본 시스템은 Raspberry Pi 기반 자율주행 로봇 플랫폼(Picar-4WD)을 중심으로 구성되어 있으며, 다음과 같은 하드웨어를 사용합니다.

- **Raspberry Pi 4 Model B (8GB)**  
  - 영상 처리 및 라인트레이싱 제어를 담당하는 중앙 제어 유닛  
  - Wi-Fi를 통해 로컬 서버와 실시간 통신

- **PiCamera v2**  
  - MJPEG 영상 스트리밍을 통해 실시간 화재·연기 감지 수행  
  - ArUco 마커 인식

- **SunFounder PiCar-4WD 키트**
  - https://docs.sunfounder.com/projects/picar-4wd/en/latest/
  - DC 모터 4개, 3채널 라인트레이서(IR 센서), 포토인터럽트 센서 (속도 측정용)
    * 초음파 센서와 서보 모터는 사용하지 않음

- **적외선 라인트레이서 센서 (3채널)**  
  - 바닥의 흑백 라인을 감지하여 경로 판단에 사용

- **DHT11 온습도 센서**  
  - 실시간 온도 및 습도 측정을 통해 환경 정보 수집  


## 🚀 핵심 기능
- **자율 주행 및 순찰**
  - 바닥 라인을 따라 주행하는 PID 기반 라인트레이싱 제어

  - 주행 중 ArUco 마커를 인식하여 순찰 경로 상의 위치를 파악하고 기록

  - API를 통해 주행 시작/정지 및 순찰 모드를 유연하게 전환 가능

- **실시간 화재·연기 감지**
  - 라즈베리파이 카메라로 전송되는 MJPEG 스트림을 프레임 단위로 수신

  - 수신된 프레임을 TFLite 모델(EfficientDet-Lite)로 분석하여 화재 또는 연기 객체 탐지

  - 동일 객체가 일정 시간 이상(1.2초) 연속으로 감지되면 경고 알림을 트리거

- **LLM을 활용한 자연어 경고 및 순찰 보고서 생성, 사용자 Q&A**
  - Upstage Solar Pro2 preview 사용
    
  - 감지된 시간, 위치, 객체 정보 등을 기반으로 LLM(Upstage API)에 상황 전달

  - 화재 감지 시 자연어 경고 메시지 생성

  - 순찰 완료 후 감지 요약과 순찰 시간, 경로가 포함된 보고서 자동 생성

  - 사용자와 실시간 질의응답 기능 (온·습도, 로봇 상태, 로봇의 위치 등)

- **Streamlit 기반 대시보드**
  - 실시간 영상 스트리밍, 마지막 탐지 이미지, 온·습도 센서 데이터 표시

  - PID 속도 조절, 탐지/주행 모드 전환 등의 제어 UI 제공

  - 사용자의 질의에 응답하는 채팅 인터페이스


## 🗂️ 디렉토리 구조
```
.
├── local/
│ ├── local_server.py # Streamlit UI + 제어 로직
│ ├── chatbot_helper.py # 화재 감지 메시지·순찰 보고서 생성
│ └── model1.tflite # EfficientDet-Lite1 화재 감지 모델
├── picar-4wd/ # Picar-4WD 제어 라이브러리
│ └── picar_4wd/ # 기본 하드웨어 제어 코드
└── examples/
  ├── pi_server.py # Raspberry Pi Flask 서버
  ├── track_line.py # Line Tracking 테스트 코드
  ├── move_forward.py # 모터 동작 테스트 코드
  └── camera_web_stream.py # Pi-cam 송출 테스트 코드
```


## 📄 주요 파일 설명

| 파일                              | 역할                                                         |
|-----------------------------------|--------------------------------------------------------------|
| `local/local_server.py`           | Streamlit 기반 대시보드 앱 (영상 처리, 탐지, 센서·모드 제어)     |
| `local/chatbot_helper.py`         | Upstage LLM 연동: 감지 알림·순찰 보고서 생성                    |
| `local/model1.tflite`             | EfficientDet-Lite1 화재·연기 탐지 TFLite 모델                  |
| `local/last_detect.jpg`           | 마지막 감지 시 프레임 저장 (실행 시 생성)                       |
| `pi_server.py`                    | Flask 서버: 카메라 스트림, PID 주행, ArUco·센서 API 제공        |
| `picar-4wd/picar_4wd/...`         | 모터·PWM 제어, 센서 인터페이스 등 로봇 제어 기능                |


## ⚙️ 설치 및 실행
1. **Picar Setup**  
    ```bash
    cd picar-4wd/
    sudo python3 setup.py install
    ```
    자세한 내용은 Picar-4wd documentation 참고:
    https://docs.sunfounder.com/projects/picar-4wd/en/latest/test_the_modules.html

3. **Raspberry Pi 서버 실행**  
    ```bash
    git clone <repo-url>
    cd <repo-root>
    pip3 install -r pi_requirements.txt
    python3 pi_server.py    # 포트 8000
    ```

4. **로컬 PC 대시보드 실행**  
    ```bash
    cd <repo-root>/local
    pip install -r local_requirements.txt
    streamlit run local_server.py  # 기본 포트 8501
    ```

5. **웹 브라우저 접속**  
    ```
    http://localhost:8501
    ```

6. **Streamlit 좌측 사이드바**  
    - Tracking Start / Stop  
    - Fire Detection Start / Stop  
    - ArUco Detection Start / Stop  
    - Base Speed 조절  

## 💡 실행 예시

### 🔍 Streamlit 대시보드
- Streamlit 기반 화재 감지 시스템 초기 화면
<img src="https://github.com/user-attachments/assets/5b499ac9-8269-4390-b69f-c01ec1625c2a" width="70%" />  

### 🔥 화재 감지 및 경고 메시지 생성
- 화재 감지 및 메시지 생성 + 질의 응답
<img src="https://github.com/user-attachments/assets/d7d08d73-97df-44e3-b8b1-3ff5ca8c100b" width="70%" />  

### 📝 탐지 로그 및 보고서
- Detected Log 예시
<img src="https://github.com/user-attachments/assets/59a7df13-dda1-4e61-a853-1f78c9285d8b" width="50%" />  

- Report 생성 결과
<img src="https://github.com/user-attachments/assets/0787dfdd-06d7-4f78-a490-69096ef90be4" width="60%" />  

## ▶️ 데모 영상
[![Video Label](http://img.youtube.com/vi/3GLE6odzigI/0.jpg)]
[(https://www.youtu.be/3GLE6odzigI)](https://www.youtube.com/watch?v=3GLE6odzigI)
