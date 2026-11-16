"""한국어 명령 → 로봇 계획 JSON 학습 데이터 생성.

실행: python make_data.py
결과: data/train.jsonl, data/val.jsonl, data/test.jsonl
- data/test.jsonl 이 이미 있으면 그대로 두고 train/val 만 새로 만든다 (모델·데이터 버전 간 같은 test 로 비교하기 위해). test 도 새로 만들려면 --new_test
- 각 줄: {"input": 한국어 명령, "output": 정답 JSON 문자열, "template": 템플릿 이름, "heldout": 처음 보는 패턴 여부}
- test 에는 학습 때 쓰지 않은 문장 템플릿(held-out)이 섞여 있어서, 외운 문장이 아닌 새 표현도 처리하는지 확인할 수 있다.
"""
import argparse
import json
import os
import random

from schema import SHELVES, SPECIAL_LOCATIONS, ITEMS, FOLLOW_TARGETS

random.seed(42)

# ── 한국어 조사 처리 ─────────────────────────────────────────────
# 숫자·영문 끝 글자를 읽었을 때 받침이 있는지 (예: 3=삼 → 받침 ㅁ, 2=이 → 없음)
_DIGIT_BATCHIM = {"0": True, "1": True, "2": False, "3": True, "4": False,
                  "5": False, "6": True, "7": True, "8": True, "9": False}
_DIGIT_RIEUL = {"1", "7", "8"}          # 일·칠·팔 → ㄹ 받침 (으로 대신 로)


def _last_info(word):
    """(받침 있음, ㄹ 받침) 반환"""
    ch = word.strip()[-1]
    if "가" <= ch <= "힣":
        jong = (ord(ch) - 0xAC00) % 28
        return jong != 0, jong == 8
    if ch.isdigit():
        return _DIGIT_BATCHIM[ch], ch in _DIGIT_RIEUL
    if ch.upper() in "LMNR":            # 엘·엠·엔·알 → 받침 있음
        return True, ch.upper() in "LR"
    return False, False


def ro(word):        # (으)로
    has, rieul = _last_info(word)
    return word + ("으로" if has and not rieul else "로")


def eul(word):       # 을/를
    return word + ("을" if _last_info(word)[0] else "를")


def ga(word):        # 이/가
    return word + ("이" if _last_info(word)[0] else "가")


# ── 위치·물건 표현 ──────────────────────────────────────────────
def shelf_phrase(loc):
    zone, num = loc[0], loc[1]
    return random.choice([
        loc, f"{zone}-{num}", f"{zone}구역 {num}번", f"{zone}열 {num}번 칸",
        f"{zone}{num}번", f"{zone}구역 {num}번 선반",
    ])


def loc_phrase(loc):
    if loc in SPECIAL_LOCATIONS:
        return random.choice(SPECIAL_LOCATIONS[loc])
    return shelf_phrase(loc)


def item_phrase(item):
    return random.choice(ITEMS[item])


def rand_shelf():
    return random.choice(SHELVES)


def rand_dest():
    return random.choice(SHELVES + ["outbound", "packing", "inbound"])


def rand_item():
    return random.choice(list(ITEMS))


# ── 계획(JSON) 조각 ──────────────────────────────────────────────
def p_go(t):
    return [{"skill": "go_to", "target": t}]


def p_go_dock(t):
    return [{"skill": "go_to", "target": t}, {"skill": "dock", "target": t}]


def p_fetch(src, item):
    return [{"skill": "go_to", "target": src}, {"skill": "dock", "target": src},
            {"skill": "load", "item": item}]


def p_deliver(dst):
    return [{"skill": "go_to", "target": dst}, {"skill": "unload"}]


def p_charge():
    return [{"skill": "go_to", "target": "charger"}, {"skill": "dock", "target": "charger"},
            {"skill": "charge"}]


# ── 단일 작업 템플릿: (문장, 계획) 을 만드는 함수들 ──────────────────
# 각 템플릿은 여러 문장 패턴을 가진다. 패턴 일부는 test 전용(held-out)으로 뺀다.
def t_move(pattern_pool):
    loc = random.choice(SHELVES + list(SPECIAL_LOCATIONS))
    lp = loc_phrase(loc)
    pats = [
        lambda: f"{ro(lp)} 가",
        lambda: f"{ro(lp)} 이동해",
        lambda: f"{lp} 쪽으로 가줘",
        lambda: f"{ro(lp)} 가 있어",
        lambda: f"지금 {ro(lp)} 이동",
        lambda: f"{lp}에 가 줘",
        lambda: f"{ro(lp)} 출발해",
        lambda: f"{lp} 쪽으로 이동 부탁해",
        lambda: f"{ro(lp)} 좀 가 봐",
        lambda: f"{lp} 앞으로 가",
        lambda: f"{ro(lp)} 향해",
        lambda: f"{lp}까지 가 줄래?",
    ]
    return pattern_pool(pats)(), p_go(loc)


def t_fetch_deliver(pattern_pool):
    src, dst, item = rand_shelf(), rand_dest(), rand_item()
    while dst == src:
        dst = rand_dest()
    s, d, i = loc_phrase(src), loc_phrase(dst), item_phrase(item)
    pats = [
        lambda: f"{s}에서 {i} 꺼내서 {d}에 갖다 놔",
        lambda: f"{s}에 있는 {eul(i)} {ro(d)} 옮겨",
        lambda: f"{s} {eul(i)} {ro(d)} 가져다줘",
        lambda: f"{i} 하나 {s}에서 실어서 {d}에 내려놔",
        lambda: f"{d}에 {i} 필요해. {s}에서 가져와",
        lambda: f"{s}의 {i} {d}에 놔 줘",
        lambda: f"{s}에서 {eul(i)} 꺼내 {ro(d)} 가져가",
        lambda: f"{d}에 {s} {i} 하나 가져다 놔",
        lambda: f"{eul(i)} {s}에서 꺼내 {d}에 둬",
        lambda: f"{s}부터 {d}까지 {i} 옮겨",
        lambda: f"{ro(d)} {s}에 있는 {i} 좀 옮겨 줘",
        lambda: f"{s}에서 {i} 싣고 {d} 가서 내려",
        lambda: f"{s}에 있는 {i}, {ro(d)} 보내",
        lambda: f"{s}에서 {d}까지 {i} 운반해",
    ]
    return pattern_pool(pats)(), p_fetch(src, item) + p_deliver(dst)


def t_fetch(pattern_pool):
    src, item = rand_shelf(), rand_item()
    s, i = loc_phrase(src), item_phrase(item)
    pats = [
        lambda: f"{s}에서 {i} 하나 실어",
        lambda: f"{s}에 가서 {eul(i)} 싣고 대기해",
        lambda: f"{s} {i} 챙겨",
        lambda: f"{s}에서 {eul(i)} 실어",
        lambda: f"{s}에서 {i} 하나 꺼내 실어",
        lambda: f"{i} 하나 {s}에서 챙겨",
        lambda: f"{s}에 들러서 {i} 실어",
        lambda: f"{s}에 있는 {i} 좀 실어 줘",
    ]
    return pattern_pool(pats)(), p_fetch(src, item)


def t_deliver(pattern_pool):
    dst = rand_dest()
    d = loc_phrase(dst)
    pats = [
        lambda: f"싣고 있는 거 {d}에 내려놔",
        lambda: f"지금 실은 짐 {ro(d)} 가져가서 내려",
        lambda: f"그거 {d}에 놓고 와",
        lambda: f"갖고 있는 거 {d}에 내려",
        lambda: f"{d}에 짐 내려놔",
        lambda: f"싣고 있는 짐 {ro(d)} 옮겨서 내려",
        lambda: f"가지고 있는 거 {d}에 두고 와",
        lambda: f"짐 {d}에 내려 줘",
    ]
    return pattern_pool(pats)(), p_deliver(dst)


def t_move_mod(pattern_pool):
    """장소를 사람·물건으로 꾸미기만 하는 이동 명령 → 정상 계획 go_to(장소) (2026-11-15).
    v3 가 '관리인이 있는 A3으로 가' 를 할 수 없는 일로 거부 → 사람이 나오는 문장 = 거부로 배우지 않게 대조 데이터.
    사건 대기('~오면')·인식 결과로 갈리는 조건('~있는지 보고 없으면')은 할 수 없는 일 (t_unsup_event·t_unsup_check)"""
    loc = random.choice(SHELVES + list(SPECIAL_LOCATIONS))
    lp, p, t = loc_phrase(loc), random.choice(PEOPLE), random.choice(THINGS)
    pats = [
        lambda: f"{ga(p)} 있는 {ro(lp)} 가",
        lambda: f"{ga(p)} 없는 {ro(lp)} 이동해",
        lambda: f"지금 {ga(p)} 있는 {ro(lp)} 가",
        lambda: f"{ga(p)} 일하고 있는 {lp} 쪽으로 가줘",
        lambda: f"{ga(p)} 기다리는 {lp}에 가 줘",
        lambda: f"{ga(t)} 놓여 있는 {ro(lp)} 가",
        lambda: f"{ga(t)} 없는 {lp} 앞으로 가",
        lambda: f"{ga(p)} 서 있는 {ro(lp)} 출발해",
        lambda: f"아까 {ga(p)} 있던 {ro(lp)} 가 봐",
        lambda: f"{ga(p)} 작업 중인 {ro(lp)} 가 볼래?",
    ]
    chosen = pattern_pool(pats)
    # '{사람}이 기다리는 {장소}' 는 사람 위치 알림(t_inform)과 같은 뜻 → go_to → dock (2026-11-16 사용자 결정).
    # 고르는 순서(무작위 호출)는 그대로라 make_scorer_sentences 의 문장은 바뀌지 않음
    return chosen(), (p_go_dock(loc) if chosen is pats[4] else p_go(loc))


def t_inform(pattern_pool):
    """사람이 있는 곳을 알려 주며 가라는 명령 → go_to(장소) → dock(장소) (2026-11-16 사용자 결정).
    사람이 그 위치에서 로봇을 기다린다 = 그리로 와서 붙으라는 뜻 (짐을 실어 줄 수 있음). 무엇을 실을지 모르므로 load 는 넣지 않음.
    마지막 패턴(시연 '관리자가 A2에서 기다립니다 가세요' 형태)은 test 전용"""
    loc = random.choice(SHELVES + list(SPECIAL_LOCATIONS))
    lp, p = loc_phrase(loc), random.choice(PEOPLE)
    pats = [
        lambda: f"{ga(p)} {lp}에서 기다리고 있어요, 가세요",
        lambda: f"{ga(p)} {lp}에 있으니 거기로 가",
        lambda: f"{lp}에서 {ga(p)} 기다리고 있어. 그쪽으로 가",
        lambda: f"{ga(p)} {lp}에 있어요. 가 주세요",
        lambda: f"{ga(p)} {lp}에서 기다리니까 거기로 이동해",
        lambda: f"{lp}에 {ga(p)} 와 있으니 가 봐",
        lambda: f"{ga(p)} {ro(lp)} 오래. 거기로 가",
        lambda: f"{ga(p)} {lp}에서 찾아. 가 봐",
        lambda: f"{ga(p)} {lp}에서 기다립니다 가세요",
    ]
    return pattern_pool(pats)(), p_go_dock(loc)


def t_charge(pattern_pool):
    pats = [
        lambda: "충전하러 가",
        lambda: "배터리 충전해",
        lambda: "가서 충전하고 있어",
        lambda: "배터리 채우러 가",
        lambda: "충전기에 붙어서 충전해",
        lambda: f"{ro(random.choice(SPECIAL_LOCATIONS['charger']))} 가서 충전해",
        lambda: "충전 좀 하고 와",
        lambda: "배터리 부족하니까 충전해",
        lambda: "충전하고 와",
        lambda: "가서 배터리 충전",
        lambda: "충전 스테이션에서 배터리 채워",
    ]
    return pattern_pool(pats)(), p_charge()


def t_follow(pattern_pool):
    tgt = random.choice(list(FOLLOW_TARGETS))
    w = random.choice(FOLLOW_TARGETS[tgt])
    if tgt == "me":
        pats = [lambda: f"{w} 따라와", lambda: f"{eul(w)} 따라와", lambda: "나만 따라오면 돼",
                lambda: "나 좀 따라와", lambda: "내 뒤로 따라와", lambda: "저 따라오세요",
                lambda: "나랑 같이 가자", lambda: "따라와 줘"]
    else:
        pats = [lambda: f"{eul(w)} 따라가", lambda: f"{w} 뒤를 따라다녀",
                lambda: f"{w} 따라가 줘", lambda: f"{w} 뒤에 붙어서 가", lambda: f"{w} 쫓아가",
                lambda: f"{w} 옆에서 같이 다녀", lambda: f"{eul(w)} 계속 따라가"]
    return pattern_pool(pats)(), [{"skill": "follow", "target": tgt}]


def t_wait(pattern_pool):
    if random.random() < 0.6:
        n = random.choice([5, 10, 15, 20, 30, 45, 60])
        txt, sec = f"{n}초", n
    else:
        n = random.choice([1, 2, 3, 5])
        txt, sec = f"{n}분", n * 60
    pats = [lambda: f"{txt} 기다려", lambda: f"{txt}만 대기해", lambda: f"거기서 {txt} 있어",
            lambda: f"{txt} 쉬어", lambda: f"{txt}만 기다려 줘", lambda: f"{txt} 대기",
            lambda: f"{txt} 동안 멈춰 있어"]
    return pattern_pool(pats)(), [{"skill": "wait", "sec": sec}]


def t_stop(pattern_pool):
    pats = [lambda: "멈춰", lambda: "정지", lambda: "스톱", lambda: "그만", lambda: "당장 멈춰",
            lambda: "정지해", lambda: "그 자리에 서", lambda: "움직이지 마", lambda: "일단 멈춰",
            lambda: "거기 서", lambda: "그만해", lambda: "멈춰 줘", lambda: "스톱해",
            lambda: "잠깐 멈춰"]
    return pattern_pool(pats)(), [{"skill": "stop"}]


SINGLE = {
    "move": t_move, "fetch_deliver": t_fetch_deliver, "fetch": t_fetch, "deliver": t_deliver,
    "charge": t_charge, "follow": t_follow, "wait": t_wait, "move_mod": t_move_mod, "inform": t_inform,
}
SINGLE_WEIGHTS = {"move": 2, "fetch_deliver": 4, "fetch": 1.5, "deliver": 1.5, "charge": 1.5,
                  "follow": 1, "wait": 1, "move_mod": 1, "inform": 1}

# 연결 표현: 앞 문장을 마침표로 끊어서 어미 활용 문제 없이 자연스럽게 잇는다
CONNECTORS = ["{a}, 그리고 {b}", "{a}. 그다음에 {b}", "{a}. 끝나면 {b}", "먼저 {a}. 그 후에 {b}",
              "{a}. 그거 하고 나서 {b}", "{a}. 이어서 {b}", "{a}. 그러고 나서 {b}", "{a}. 다음은 {b}"]
HELDOUT_CONNECTORS = ["{a}. 다 하면 {b}"]
# 두 작업 연결에서 앞에 올 수 없는 작업 (follow 는 계속 이어지는 동작이라 다음 작업으로 넘어갈 시점이 없음)
NO_FIRST = {"follow"}


def _ends_imperative(s):
    return s.rstrip("?.")


# ── 거부 데이터 ─────────────────────────────────────────────────
UNKNOWN_PLACES = ([f"{z}{n}" for z in "EFGH" for n in range(1, 6)]
                  + [f"{z}{n}" for z in "ABCD" for n in range(5, 10)]
                  + ["2층 사무실", "주차장", "옥상", "휴게실", "식당", "정문"])


def t_unknown_location(pattern_pool):
    bad = random.choice(UNKNOWN_PLACES)
    pats = [lambda: f"{ro(bad)} 가", lambda: f"{bad}에서 상자 꺼내서 출고장에 갖다 놔",
            lambda: f"{ro(bad)} 이동해", lambda: f"{bad}에 가 있어", lambda: f"{bad} 쪽으로 가줘",
            lambda: f"{bad}에서 바구니 실어", lambda: f"박스 {ro(bad)} 옮겨",
            lambda: f"택배 {bad}에 내려놔"]
    return pattern_pool(pats)(), None, "unknown_location"


def t_unknown_mod(pattern_pool):
    """꾸밈말·사람 위치 알림 + 목록에 없는 위치 → unknown_location (2026-11-16, v5).
    v4 에서 꾸밈말을 넣은 뒤 해석 모델만 쓸 때 목록 밖 장소 잘못 실행이 27 → 83 으로 늘어서 되돌리려는 거부 예시.
    없는 위치는 t_unknown_location 과 같은 목록 (평가용 장소는 쓰지 않음). 마지막 패턴은 test 전용"""
    bad = random.choice(UNKNOWN_PLACES)
    p, t = random.choice(PEOPLE), random.choice(THINGS)
    pats = [
        lambda: f"{ga(p)} 있는 {ro(bad)} 가",
        lambda: f"{ga(p)} 없는 {ro(bad)} 이동해",
        lambda: f"{ga(t)} 놓여 있는 {ro(bad)} 가",
        lambda: f"{ga(p)} 일하고 있는 {bad} 쪽으로 가줘",
        lambda: f"{ga(p)} 기다리는 {bad}에 가 줘",
        lambda: f"{ga(p)} {bad}에서 기다리고 있어요, 가세요",
        lambda: f"{ga(p)} {bad}에 있으니 거기로 가",
        lambda: f"{bad}에 {ga(p)} 와 있으니 가 봐",
        lambda: f"{ga(p)} {bad}에서 기다립니다 가세요",
    ]
    return pattern_pool(pats)(), None, "unknown_location"


def t_unsupported(pattern_pool):
    acts = ["춤춰", "노래 불러", "커피 타 와", "문 열어", "사진 찍어", "날씨 알려줘", "청소해",
            "계단 올라가", "상자 열어 봐", "불 꺼", "점프해", "전화 걸어", "상자 안에 뭐 있는지 봐",
            "엘리베이터 타", "창문 닫아", "음악 틀어", "택배 뜯어", "물 갖다줘", "농담 해 봐",
            "재고 몇 개인지 세어 봐", "트럭에 실어", "바닥 닦아", "사람 불러 와", "회의 잡아 줘"]
    pats = [lambda: random.choice(acts), lambda: random.choice(acts) + " 줘",
            lambda: "지금 " + random.choice(acts), lambda: random.choice(acts) + " 좀"]
    return pattern_pool(pats)(), None, "unsupported_command"


# 할 수 없는 일 추가 유형 (2026-11-15, 시연에서 발견: '충전기가져와' → 충전하러 감,
# '화장실에서 관리자가 나오면 관리자보다 먼저 A1으로 가' → 2B 가 할 수 없는 일로 보지 않음).
# 규칙·장소 목록 대조 없이 데이터로 해결. 각 유형의 마지막 패턴은 test 전용(held-out)
NON_ITEMS = ["충전기", "충전 스테이션", "선반", "포장대", "작업대", "입고대", "지게차", "컨베이어", "랙", "책상"]
PEOPLE = ["관리자", "작업자", "팀장님", "사람", "기사님", "다른 로봇", "직원", "반장님", "경비원"]   # 뒤 3개 2026-11-15 추가
THINGS = ["지게차", "상자", "택배", "빈 팔레트", "카트"]
ROOMS = ["화장실", "사무실", "회의실", "휴게실", "엘리베이터", "탈의실"]
# '관리인' 은 일부러 넣지 않음: 시연 문장('관리인이 있는 A3으로 가' 등)을 처음 보는 단어로 평가


def _any_loc():
    return loc_phrase(random.choice(SHELVES + list(SPECIAL_LOCATIONS)))


def t_unsup_object(pattern_pool):
    """물건이 아닌 대상(설비·가구)을 운반하라는 명령. 로봇이 싣는 물건은 상자·택배·바구니뿐"""
    obj = random.choice(NON_ITEMS)
    lp = _any_loc()
    while obj in lp or lp in obj:              # '포장대를 포장대로 옮겨' 같은 문장 제외
        lp = _any_loc()
    pats = [
        lambda: f"{obj} 가져와",
        lambda: f"{obj} 들고 와",
        lambda: f"{eul(obj)} {ro(lp)} 옮겨",
        lambda: f"{obj} 좀 가져다줘",
        lambda: f"{lp}에 {obj} 갖다 놔",
        lambda: f"{obj} 싣고 와",
        lambda: f"{obj.replace(' ', '')}{random.choice(['가져와', '들고와', '갖다줘'])}",   # 띄어쓰기 없이 입력
        lambda: f"{obj} 하나 가져와",
        lambda: f"{obj} 옮겨 와",
    ]
    return pattern_pool(pats)(), None, "unsupported_command"


def t_unsup_event(pattern_pool):
    """조건·사건을 기다렸다가 움직이라는 명령. 로봇은 사건을 감지해 기다리는 기능이 없음 (wait 는 정해진 시간만)"""
    who = random.choice(PEOPLE)
    event = random.choice([f"{ga(who)} 오면", f"{ga(who)} 나오면", f"{ga(who)} 부르면", "트럭이 도착하면",
                           "문이 열리면", "택배가 들어오면", "벨이 울리면", "점심시간 되면", "3시가 되면", "불이 켜지면"])
    room, lp = random.choice(ROOMS), _any_loc()
    pats = [
        lambda: f"{event} {ro(lp)} 가",
        lambda: f"{event} 출발해",
        lambda: f"{room}에서 {ga(who)} 나오면 {ro(lp)} 가",
        lambda: f"{event} {ro(lp)} 이동해",
        lambda: f"{event} 바로 {lp}에 가 있어",
        lambda: f"{event} 충전하러 가",
        lambda: f"{event} 알려 줘",
        lambda: f"{lp}에서 기다렸다가 {event} 출발해",
    ]
    return pattern_pool(pats)(), None, "unsupported_command"


def t_unsup_race(pattern_pool):
    """사람·다른 기계와 경쟁·비교하라는 명령 ('~보다 먼저 가'). 연결 표현 '먼저 A. 그 후에 B' 와는 다름"""
    who = random.choice(["관리자", "작업자", "저 사람", "다른 로봇", "지게차", "팀장님"])
    lp = _any_loc()
    pats = [
        lambda: f"{who}보다 먼저 {ro(lp)} 가",
        lambda: f"{who}보다 빨리 {lp}에 가",
        lambda: f"{who}하고 {lp}까지 시합해",
        lambda: f"{eul(who)} 앞질러서 {ro(lp)} 가",
        lambda: f"{who}보다 먼저 {lp}에 도착해",
        lambda: f"{ga(who)} 가기 전에 {ro(lp)} 가",
        lambda: f"{lp}까지 {who}보다 빨리 가",
    ]
    return pattern_pool(pats)(), None, "unsupported_command"


def t_unsup_check(pattern_pool):
    """사람·물건이 있는지 알아본 결과에 따라 행동이 갈리는 명령 (2026-11-15). 로봇은 사람·물건을 알아보는 기능이 없음.
    '{사람}이 있는 {장소}로 가' 처럼 장소를 꾸미기만 하는 문장(t_move_mod)은 정상 명령"""
    who = random.choice(PEOPLE + THINGS)
    lp, lp2 = _any_loc(), _any_loc()
    while lp2 == lp:
        lp2 = _any_loc()
    pats = [
        lambda: f"{lp}에 {ga(who)} 있는지 보고 없으면 {ro(lp2)} 가",
        lambda: f"{ga(who)} {lp}에 있으면 {ro(lp2)} 가",
        lambda: f"{lp}에 {ga(who)} 없으면 {ro(lp2)} 이동해",
        lambda: f"{lp} 가서 {ga(who)} 있는지 확인해",
        lambda: f"{ga(who)} 있는지 확인하고 없으면 {ro(lp)} 가",
        lambda: f"{lp}에 {ga(who)} 있으면 거기서 기다려",
        lambda: f"{ga(who)} 없을 때 {ro(lp)} 가",
        lambda: f"{ga(who)} {lp}에 있는지 가서 확인하고 없다면 {ro(lp2)} 가",
    ]
    return pattern_pool(pats)(), None, "unsupported_command"


def t_unsup_wait(pattern_pool):
    """위치 알림('~가 ~에서 기다리고 있어요, 가세요' → go_to → dock)과 구별하는 대조 문장 (2026-11-16, v6).
    v5 에서 위치 알림 데이터가 처음 보는 '~에서 기다렸다가 ~가 부르면 출발해'·'~가 ~에 있는지 가서 확인하고 없다면 ~로 가'
    로 번져 go_to → dock 으로 해석됨 → 로봇이 그 자리에서 기다리다가 사건이 생기면 움직이기 / 가서 사람이 있는지 보고 오기 는
    할 수 없는 일. 주의: 평가 전용 패턴과 구조가 비슷해서 그 패턴의 점수가 실제보다 좋게 나올 수 있음"""
    who = random.choice(PEOPLE)
    event = random.choice([f"{ga(who)} 오면", f"{ga(who)} 나오면", f"{ga(who)} 부르면", "트럭이 도착하면",
                           "문이 열리면", "택배가 들어오면", "벨이 울리면", "점심시간 되면", "3시가 되면", "불이 켜지면"])
    lp, lp2 = _any_loc(), _any_loc()
    while lp2 == lp:
        lp2 = _any_loc()
    pats = [
        lambda: f"{lp}에서 대기하다가 {event} 가",
        lambda: f"{lp}에 있다가 {event} {ro(lp2)} 가",
        lambda: f"{lp}에서 기다리다가 {event} {ro(lp2)} 이동해",
        lambda: f"{lp}에 서 있다가 {event} 출발해",
        lambda: f"{lp}에 가서 {ga(who)} 있는지 보고 와",
        lambda: f"{lp}에 가서 {ga(who)} 있는지 보고 없으면 {ro(lp2)} 가",
        lambda: f"{lp} 가서 {who} 있나 확인하고 와",
        lambda: f"{lp}에 가서 {ga(who)} 있으면 거기서 기다려",
        lambda: f"{lp}에서 멈춰 있다가 {event} 움직여",
    ]
    return pattern_pool(pats)(), None, "unsupported_command"


UNSUP_EXTRA = {"unsup_object": t_unsup_object, "unsup_event": t_unsup_event, "unsup_race": t_unsup_race,
               "unsup_check": t_unsup_check, "unsup_wait": t_unsup_wait}


# ── 템플릿 패턴 분리: 각 템플릿의 마지막 패턴은 test 전용 ────────────────
def make_pool(mode):
    """mode='train': 마지막 패턴 제외 / 'test_heldout': 마지막 패턴만 / 'any': 전부"""
    def pool(pats):
        if mode == "train":
            return random.choice(pats[:-1]) if len(pats) > 1 else pats[0]
        if mode == "heldout":
            return pats[-1]
        return random.choice(pats)
    return pool


def sample(mode):
    pool = make_pool(mode)
    r = random.random()
    if r < 0.10:
        txt, _, err = t_unknown_location(pool)
        return txt, {"plan": [], "error": err}, "unknown_location"
    if r < 0.15:
        txt, _, err = t_unsupported(pool)
        return txt, {"plan": [], "error": err}, "unsupported"
    if r < 0.25:                                   # 할 수 없는 일 추가 유형 5가지 (2026-11-15~16, 각 2%)
        name = random.choice(list(UNSUP_EXTRA))
        txt, _, err = UNSUP_EXTRA[name](pool)
        return txt, {"plan": [], "error": err}, name
    if r < 0.27:                                   # 꾸밈말·위치 알림 + 없는 위치 (2026-11-16, v5, 2%)
        txt, _, err = t_unknown_mod(pool)
        return txt, {"plan": [], "error": err}, "unknown_mod"
    if r < 0.32:
        txt, plan = t_stop(pool)
        return txt, {"plan": plan}, "stop"
    names = list(SINGLE)
    weights = [SINGLE_WEIGHTS[n] for n in names]
    if r < 0.52:                                   # 두 작업 연결
        n1, n2 = random.choices(names, weights, k=2)
        while n1 in NO_FIRST or n1 == n2:          # 같은 작업 두 번(충전→충전 등)이나 follow 다음 작업은 제외
            n1, n2 = random.choices(names, weights, k=2)
        a, pa = SINGLE[n1](pool)
        b, pb = SINGLE[n2](pool)
        conns = HELDOUT_CONNECTORS if mode == "heldout" else CONNECTORS
        txt = random.choice(conns).format(a=_ends_imperative(a), b=b)
        return txt, {"plan": pa + pb}, f"{n1}+{n2}"
    n = random.choices(names, weights)[0]
    txt, plan = SINGLE[n](pool)
    return txt, {"plan": plan}, n


def build(n, mode, owner, split):
    """owner: 문장 → 처음 나온 split 이름. 다른 split 에 이미 나온 문장은 버리고,
    같은 split 안에서는 같은 문장을 최대 3번까지 허용 (짧은 명령은 표현 수가 적어서)"""
    rows, count = [], {}
    tries = 0
    while len(rows) < n and tries < n * 50:
        tries += 1
        txt, out, tmpl = sample(mode)
        if owner.get(txt, split) != split or count.get(txt, 0) >= 3:
            continue
        owner[txt] = split
        count[txt] = count.get(txt, 0) + 1
        rows.append({"input": txt, "output": json.dumps(out, ensure_ascii=False), "template": tmpl,
                     "heldout": mode == "heldout"})   # True = 학습 때 못 본 문장 패턴
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new_test", action="store_true",
                    help="test 도 새로 만듦 (기본: 기존 data/test.jsonl 을 그대로 써서 모델 간 비교가 공정하도록 유지)")
    args = ap.parse_args()

    os.makedirs("data", exist_ok=True)
    owner = {}
    test = None
    if os.path.exists("data/test.jsonl") and not args.new_test:
        test = [json.loads(l) for l in open("data/test.jsonl", encoding="utf-8")]
        for row in test:
            owner[row["input"]] = "test"            # 기존 test 문장은 train/val 에 들어가지 않게 예약
        print("기존 test 유지 (data/test.jsonl)")
    if os.path.exists("data/probe_unsupported.jsonl"):     # 추가 평가 세트 문장도 train/val 에 들어가지 않게 예약 (2026-11-16, v6)
        n0 = len(owner)
        for line in open("data/probe_unsupported.jsonl", encoding="utf-8"):
            owner.setdefault(json.loads(line)["input"], "probe")
        print(f"추가 평가 세트 문장 {len(owner) - n0}개 예약 (data/probe_unsupported.jsonl)")
    train = build(4000, "train", owner, "train")
    val = build(300, "train", owner, "val")
    if test is None:
        test = build(200, "train", owner, "test") + build(200, "heldout", owner, "test")   # 절반은 처음 보는 문장 패턴
    for name, rows in [("train", train), ("val", val), ("test", test)]:
        with open(f"data/{name}.jsonl", "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)}개")
    print("\n예시:")
    for row in random.sample(train, 8):
        print(" ", row["input"], "→", row["output"])


if __name__ == "__main__":
    main()
