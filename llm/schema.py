"""창고 로봇 명령 체계 (기술 목록, 위치, 물건, 시스템 프롬프트) — 데이터 생성·학습·평가가 모두 이 파일을 기준으로 한다."""

# 선반 위치: A~D 구역 × 1~4 칸
SHELVES = [f"{zone}{num}" for zone in "ABCD" for num in range(1, 5)]

# 특수 위치: JSON 에 쓰는 이름 → 한국어 표현들
SPECIAL_LOCATIONS = {
    "inbound": ["입고장", "입고 구역", "입고대"],
    "outbound": ["출고장", "출고 구역", "출하장"],
    "packing": ["포장대", "포장 구역", "포장하는 곳"],
    "charger": ["충전소", "충전기", "충전 스테이션"],
}

LOCATIONS = SHELVES + list(SPECIAL_LOCATIONS)

# 물건: JSON 이름 → 한국어 표현들
ITEMS = {
    "box": ["상자", "박스"],
    "parcel": ["택배", "소포"],
    "tote": ["바구니", "토트"],
}

# 따라갈 대상
FOLLOW_TARGETS = {
    "me": ["나", "저"],
    "worker": ["작업자", "저 작업자", "작업자분"],
}

# 기술 목록 (강화학습: go_to, dock, follow / 스크립트: load, unload, charge, wait, stop)
SKILLS = ["go_to", "dock", "follow", "load", "unload", "charge", "wait", "stop"]

# 거부할 때 쓰는 에러 코드
ERRORS = ["unknown_location", "unsupported_command"]

SYSTEM_PROMPT = (
    "너는 창고 운반 로봇의 명령 해석기다. 사용자의 한국어 명령을 JSON 계획으로만 답한다.\n"
    "위치: A1~D4 선반, inbound(입고장), outbound(출고장), packing(포장대), charger(충전소)\n"
    "물건: box(상자), parcel(택배), tote(바구니)\n"
    "기술: go_to(target), dock(target), follow(target: me|worker), load(item), unload, charge, wait(sec), stop\n"
    '형식: {"plan": [{"skill": ..., ...}]}\n'
    '없는 위치면 {"plan": [], "error": "unknown_location"}, 할 수 없는 일이면 {"plan": [], "error": "unsupported_command"}'
)
