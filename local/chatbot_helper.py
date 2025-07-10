import os
from langchain_upstage import ChatUpstage
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from dotenv import load_dotenv

class DisasterMessageGenerator:
    # 영어→한글 변환 딕셔너리
    DETECTED_TYPE_KOR = {
        "fire": "화재",
        "smoke": "연기"
    }

    def __init__(self):
        load_dotenv()
        if not os.getenv("UPSTAGE_API_KEY"):
            raise ValueError("UPSTAGE_API_KEY is not set in the environment variables.")
        self.llm = ChatUpstage(model_name="solar-pro2-preview")
        self.prompt_template = PromptTemplate(
            input_variables=["time", "aruco_id", "box_width", "box_height", "confidence", "detected_object_type"],
            template="""
당신은 화재 감시 로봇을 보조하는 친절하고 유능한 AI 어시스턴트입니다. 로봇이 {detected_object_type}으로 의심되는 상황을 포착했습니다. 아래 데이터를 바탕으로 상황의 위험도를 평가하고, 자연스럽고 상세한 한국어 메시지를 작성해주세요.

[감지 정보]
- 시간: {time}
- 위치: 구간 {aruco_id}
- 감지된 객체 크기: 가로 {box_width}px, 세로 {box_height}px
- {detected_object_type} 예측 확률: {confidence:.2f}

[작성 가이드]
1.  **위험도 판단:** 예측 확률과 객체 크기를 종합적으로 고려하여 위험도를 '관심', '주의', '경계', '심각' 단계로 판단해주세요.
2.  **메시지 스타일:** 딱딱한 보고서가 아닌, 상황의 심각성을 명확히 전달하면서도 자연스러운 대화체로 작성해주세요.
3.  **핵심 내용 포함:**
    - 메시지 시작은 항상 '[{detected_object_type} 감지 알림]'으로 해주세요.
    - 언제, 어디서, 무엇을 감지했는지 명확히 설명해주세요.
    - 직접 판단한 위험도 단계를 언급하고, 왜 그렇게 판단했는지 간략히 설명해주세요. (예: "확률이 높고 객체가 커서 '경계' 단계로 판단됩니다.")
    - 관리자가 어떤 조치를 취해야 할지 추천해주세요.
    - 최소 3문장 이상으로 상세하게 작성해주세요.
    - 공식적인 언어를 사용하되, 너무 딱딱하지 않게 자연스럽게 작성해주세요.

상황의 시급함이 느껴지면서도, 신뢰감을 주는 메시지를 생성해주세요.
"""
        )
        self.patrol_report_template = PromptTemplate(
            input_variables=["patrolled_markers_log", "patrol_events", "start_time", "end_time", "num_markers", "marker_ids"],
            template="""
당신은 로봇 화재 탐지 시스템의 AI입니다. 로봇이 순찰 임무를 한 바퀴 완료했습니다. 아래 데이터를 바탕으로 사실에 기반한 순찰 결과 보고서를 한국어로 작성해주세요.

[순찰 데이터]
- 통과 구간 로그 (시간, ArUco 마커 ID): 
{patrolled_markers_log}
- 순찰 중 발생한 주요 이벤트:
{patrol_events}

[보고서 작성 가이드]
1.  **제목:** 보고서는 항상 `[화재 순찰 결과]', 줄바꿈 두 번 후 '로봇이 순찰을 완료했습니다.` 로 시작해야 합니다.
2.  **내용:**
    - 순찰 시작 시간({start_time})과 종료 시간({end_time}), 그리고 전체 경로({marker_ids})을 기술합니다.
    - 순찰 중 통과한 ArUco 마커 ID와 시간 정보를 나열합니다.
    - 이벤트가 있다면, 각 이벤트를 발생 순서대로 객관적으로 나열합니다.
3.  **규칙:**
    - 보고서 마지막에 "...큰 도움이 될 것입니다." 와 같은 주관적인 평가, 의견을 덧붙이지 마세요.
    - 제공된 데이터에 없는 내용을 추측하거나 생성하지 마세요.

지금 바로 지시사항에 따라 순찰 결과 보고서를 작성하세요.
"""
        )

    def generate_message(self, time: str, aruco_id: str, box_width: int, box_height: int, score: float, detected_object_type: str) -> str:
        display_location = f"구간 {aruco_id}"
        if str(aruco_id) == "190":
            display_location = "화재 감지 지점 (위치 특정 불가)"
        
        # 2. 한글 변환 적용
        detected_object_type_kr = self.DETECTED_TYPE_KOR.get(detected_object_type, detected_object_type)
        
        prompt = self.prompt_template.format_prompt(
            time=time,
            aruco_id=display_location,
            box_width=box_width,
            box_height=box_height,
            confidence=score,
            detected_object_type=detected_object_type_kr   # 한글값 전달!
        ).to_string()
        try:
            message = self.llm.invoke([HumanMessage(content=prompt)])
            return message.content
        except Exception as e:
            print(f"[ERROR] Failed to generate message from LLM: {e}")
            return "긴급: 화재가 감지되었습니다. 즉시 확인 바랍니다. (AI 모델 호출 오류)"

    def generate_patrol_report(self, patrolled_markers: list, patrol_events: list) -> str:
        if not patrolled_markers:
            return "[순찰 보고서 생성 오류] 순찰 기록이 없습니다."

        event_summary = "- 특이사항 없음" if not patrol_events else "\n".join([f"- {event}" for event in patrol_events])
        markers_log_str = "\n".join(
            [f"- {timestamp}: 구간 {marker_id} 통과" for timestamp, marker_id in patrolled_markers if str(marker_id) != "190"]
        )

        start_time = patrolled_markers[0][0]
        end_time = patrolled_markers[-1][0]
        marker_ids = ", ".join([str(m_id) for t, m_id in patrolled_markers if str(m_id) != "190"])

        prompt = self.patrol_report_template.format_prompt(
            patrolled_markers_log=markers_log_str,
            patrol_events=event_summary,
            start_time=start_time,
            end_time=end_time,
            num_markers=len(patrolled_markers),
            marker_ids=marker_ids
        ).to_string()
        try:
            message = self.llm.invoke([HumanMessage(content=prompt)])
            return message.content
        except Exception as e:
            print(f"[ERROR] Failed to generate patrol report from LLM: {e}")
            return "[순찰 보고서 생성 오류] AI 모델을 호출하는 데 실패했습니다."

    def answer_question(self, question: str, current_status: dict) -> str:
        def get_detection_details():
            latest = current_status.get('latest_detection_details')
            if not latest:
                return "- 최근 감지된 화재/연기 정보가 없습니다."
            try:
                return (
                    f"- 감지 시간: {latest.get('time', '알 수 없음')}\n"
                    f"- 감지된 객체: {latest.get('class_name', '알 수 없음')}\n"
                    f"- 예측 확률: {latest.get('score', 0):.2f}\n"
                    f"- 감지된 객체 크기: 가로 {latest['box'][2] - latest['box'][0]}px, 세로 {latest['box'][3] - latest['box'][1]}px"
                )
            except Exception:
                return "- 최근 감지된 화재/연기 정보가 없습니다."

        prompt = f"""
당신은 로봇 관제 시스템의 AI입니다. 사용자의 질문에 대해 아래 로봇의 최신 상태 정보를 바탕으로, 친절하고 유능하게 답변해주세요. 특히 최근 감지된 화재나 연기 정보가 있다면, 이를 바탕으로 상황을 설명하고 필요한 경우 추가적인 정보를 제공할 수 있습니다.

[로봇 현재 상태]
- 현재 시간: {current_status.get('time', '알 수 없음')}
- 로봇 위치: {"화재 감지 지점 (위치 특정 불가)" if str(current_status.get('last_detected_aruco_id')) == "190" else (f"아루코 마커 {current_status.get('last_detected_aruco_id')} 근처" if current_status.get('last_detected_aruco_id') else "아직 특정되지 않음")}
- 주변 온도: {f"{current_status.get('temperature')}°C" if current_status.get('temperature') is not None else "측정되지 않음"}
- 주변 습도: {f"{current_status.get('humidity')}%" if current_status.get('humidity') is not None else "측정되지 않음"}
- 추적 모드: {"활성화" if current_status.get('tracking_enabled') else "비활성화"}
- 화재 감지 모드: {"활성화" if current_status.get('detection_enabled') else "비활성화"}
- 위치 추적(ArUco) 모드: {"활성화" if current_status.get('aruco_enabled') else "비활성화"}

[최근 감지 정보]
{current_status.get('latest_disaster_message') or get_detection_details()}

[사용자 질문]
"{question}"

[답변 가이드]
- 제공된 '로봇 현재 상태'와 '최근 감지 정보'만을 사용하여 답변하세요.
- 사용자가 '지금', '현재'와 같은 단어로 현재 상태를 물으면 '로봇 현재 상태' 섹션의 정보를 우선적으로 활용하여 답변하세요.
- 사용자가 '이전', '과거', '최근 감지 알림'과 같은 단어로 이전 정보를 물으면 '최근 감지 정보' 섹션의 'latest_disaster_message' 내용을 우선적으로 활용하여 답변하세요. 만약 'latest_disaster_message'가 없다면 'get_detection_details()'의 내용을 활용하세요.
- "준비하고 있습니다", "할 수 있습니다" 와 같은 추측성, 수동적 표현이나 미래에 대한 예측을 절대 사용하지 마세요.
- 로봇의 상태를 해석하거나 부연 설명하지 말고, 있는 그대로의 사실만 전달하세요. 
- 불필요한 미사여구는 제외하고, 질문에 대한 핵심 정보만 간결하게 전달하세요.
"""

        try:
            message = self.llm.invoke([HumanMessage(content=prompt)])
            return message.content
        except Exception as e:
            print(f"[ERROR] Failed to answer question from LLM: {e}")
            return "죄송합니다. 질문을 처리하는 중에 오류가 발생했습니다. 다시 시도해주세요."
