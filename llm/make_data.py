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
    "charge": t_charge, "follow": t_follow, "wait": t_wait,
}
SINGLE_WEIGHTS = {"move": 2, "fetch_deliver": 4, "fetch": 1.5, "deliver": 1.5, "charge": 1.5,
                  "follow": 1, "wait": 1}

# 연결 표현: 앞 문장을 마침표로 끊어서 어미 활용 문제 없이 자연스럽게 잇는다
CONNECTORS = ["{a}, 그리고 {b}", "{a}. 그다음에 {b}", "{a}. 끝나면 {b}", "먼저 {a}. 그 후에 {b}",
              "{a}. 그거 하고 나서 {b}", "{a}. 이어서 {b}", "{a}. 그러고 나서 {b}", "{a}. 다음은 {b}"]
HELDOUT_CONNECTORS = ["{a}. 다 하면 {b}"]
# 두 작업 연결에서 앞에 올 수 없는 작업 (follow 는 계속 이어지는 동작이라 다음 작업으로 넘어갈 시점이 없음)
NO_FIRST = {"follow"}


def _ends_imperative(s):
    return s.rstrip("?.")


# ── 거부 데이터 ─────────────────────────────────────────────────
def t_unknown_location(pattern_pool):
    bad = random.choice(
        [f"{z}{n}" for z in "EFGH" for n in range(1, 6)]
        + [f"{z}{n}" for z in "ABCD" for n in range(5, 10)]
        + ["2층 사무실", "주차장", "옥상", "휴게실", "식당", "정문"]
    )
    pats = [lambda: f"{ro(bad)} 가", lambda: f"{bad}에서 상자 꺼내서 출고장에 갖다 놔",
            lambda: f"{ro(bad)} 이동해", lambda: f"{bad}에 가 있어", lambda: f"{bad} 쪽으로 가줘",
            lambda: f"{bad}에서 바구니 실어", lambda: f"박스 {ro(bad)} 옮겨",
            lambda: f"택배 {bad}에 내려놔"]
    return pattern_pool(pats)(), None, "unknown_location"


def t_unsupported(pattern_pool):
    acts = ["춤춰", "노래 불러", "커피 타 와", "문 열어", "사진 찍어", "날씨 알려줘", "청소해",
            "계단 올라가", "상자 열어 봐", "불 꺼", "점프해", "전화 걸어", "상자 안에 뭐 있는지 봐",
            "엘리베이터 타", "창문 닫아", "음악 틀어", "택배 뜯어", "물 갖다줘", "농담 해 봐",
            "재고 몇 개인지 세어 봐", "트럭에 실어", "바닥 닦아", "사람 불러 와", "회의 잡아 줘"]
    pats = [lambda: random.choice(acts), lambda: random.choice(acts) + " 줘",
            lambda: "지금 " + random.choice(acts), lambda: random.choice(acts) + " 좀"]
    return pattern_pool(pats)(), None, "unsupported_command"


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
    if r < 0.20:
        txt, plan = t_stop(pool)
        return txt, {"plan": plan}, "stop"
    names = list(SINGLE)
    weights = [SINGLE_WEIGHTS[n] for n in names]
    if r < 0.40:                                   # 두 작업 연결
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
