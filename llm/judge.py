"""'확인' 판단기: 2B 해석 모델 + 점수 모델이 '확인' 을 낸 명령을 메인 PC 의 큰 모델(Ollama qwen3.5:9b)이 실행·거부로 판단.

결정 (2026-11-15 사용자): 현장에서 사람이 매번 답할 수 없다고 가정 → 네/아니요를 묻지 않고 9B 가 판단
운용: 9B 를 계속 올려 둠 (keep_alive -1), 컨텍스트 4096.
  - 모든 요청에 num_ctx 4096 을 넣어야 함. 빠뜨리면 Ollama 가 기본값(32768)으로 다시 불러옴 (약 18~19초)
판단 대상: 실행할 수 있는 계획(오류 없음, 위치가 모두 목록 안)이 있을 때만. 계획이 없으면 9B 를 부르지 않고 거부
사고(think)는 끔 (2026-11-15 시험 후 폐기: 응답 약 55초로 너무 느림)
"""
import json
import time
import urllib.request

from schema import LOCATIONS

OLLAMA = "http://127.0.0.1:11434"
MODEL = "qwen3.5:9b"
OPTIONS = {"num_ctx": 4096, "temperature": 0}

SYSTEM = """너는 창고 운반 로봇의 실행 판단기다. 작은 해석 모델이 사용자의 한국어 명령을 로봇 계획으로 바꿨는데, 확실하지 않아서 너에게 실행 여부를 묻는다. 사람에게 다시 물어볼 수 없으므로 '실행' 또는 '거부' 중 하나로만 답한다.

로봇이 갈 수 있는 위치 (이것뿐이다)
- 선반 A1~D4: A~D 구역의 1~4번 칸. 'B3', 'B-3', 'B3번', 'B구역 3번', 'B열 3번 칸', 'B구역 3번 선반' 은 모두 같은 B3 이다
- inbound: 입고장 (입고 구역, 입고대)
- outbound: 출고장 (출고 구역, 출하장)
- packing: 포장대 (포장 구역)
- charger: 로봇 전용 충전소 (충전기, 충전 스테이션). 지게차·차량 충전 설비나 배터리 설비와는 다른 곳이다
로봇이 할 수 있는 일: 이동(go_to), 위치에 정확히 붙기(dock), 따라가기(follow), 물건 싣기(load: 상자·택배·바구니), 내리기(unload), 충전(charge), 대기(wait), 정지(stop)
계획을 만드는 규칙 (이렇게 만든 계획은 정상이다)
- 선반에서 물건 싣기: go_to → dock → load (선반에 붙으면 자동으로 실림, 로봇 팔 없음)
- 어디에 갖다 놓기·내리기: go_to → unload
- 충전: go_to(charger) → dock → charge (장소를 말하지 않아도 충전소로 감)
- '거기서·여기서 기다려': 지금 자리에서 wait
- 두 작업을 이어서 말하면 순서대로 이어 붙임

실행: 사용자가 말한 장소가 계획의 위치와 같은 곳(같은 곳을 다른 이름이나 설명으로 부른 것)이 분명하고, 계획의 동작이 명령의 뜻과 맞을 때
거부: 아래 중 하나라도 해당할 때
- 사용자가 말한 장소가 목록에 없는 다른 곳이다 (기능이 비슷해도 다른 장소면 거부)
- 어느 위치를 말하는지 분명하지 않다
- 계획의 동작이 명령의 뜻과 다르다 (예: 물건을 가져오라는데 그 장소에 가서 충전함)
- 로봇이 가면 위험한 곳이다 (차량 통행로, 설비, 다른 층, 계단 등)

참고 점수 (작은 점수 모델의 예측, 0~1): 관계율 = 창고 업무와의 관련도, 위험률 = 잘못 갔을 때의 피해 크기
reason 은 한국어 한 문장으로, 무엇이 분명하고 무엇이 불확실한지 말한다."""

SCHEMA = {"type": "object",
          "properties": {"reason": {"type": "string"}, "decision": {"type": "string", "enum": ["실행", "거부"]}},
          "required": ["reason", "decision"]}


def executable(out):
    """오류 없이 계획이 있고, 계획 속 위치가 모두 목록 안에 있으면 True"""
    if not isinstance(out, dict) or out.get("error") or not isinstance(out.get("plan"), list) or not out["plan"]:
        return False
    steps = out["plan"]
    return all(isinstance(s, dict) for s in steps) and \
        all(s.get("target") in LOCATIONS for s in steps if s.get("skill") in ("go_to", "dock"))


class Judge:
    def __init__(self, url=OLLAMA, model=MODEL):
        self.url, self.model = url, model

    def _post(self, path, body, timeout=120):
        req = urllib.request.Request(self.url + path, json.dumps(body, ensure_ascii=False).encode("utf-8"),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def load(self):
        """9B 를 컨텍스트 4096 으로 올리고 계속 유지 (이미 4096 으로 올라가 있으면 바로 끝남)"""
        self._post("/api/generate", {"model": self.model, "prompt": "", "keep_alive": -1, "options": OPTIONS},
                   timeout=300)

    def unload(self):
        self._post("/api/generate", {"model": self.model, "prompt": "", "keep_alive": 0})

    def __call__(self, text, plan, rel, risk):
        """{"decision": "실행"|"거부", "reason", "latency_ms"}"""
        t0 = time.time()
        user = (f"명령: {text}\n해석 모델 계획: {json.dumps({'plan': plan}, ensure_ascii=False)}\n"
                f"점수: 관계율 {rel:.2f}, 위험률 {risk:.2f}")
        r = self._post("/api/chat", {"model": self.model, "stream": False, "think": False, "keep_alive": -1,
                                     "format": SCHEMA, "options": OPTIONS,
                                     "messages": [{"role": "system", "content": SYSTEM},
                                                  {"role": "user", "content": user}]})
        out = json.loads(r["message"]["content"])
        decision = out.get("decision") if out.get("decision") in ("실행", "거부") else "거부"
        return {"decision": decision, "reason": str(out.get("reason", "")).strip(),
                "latency_ms": int((time.time() - t0) * 1000)}
