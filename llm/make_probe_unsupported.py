"""할 수 없는 일 추가 평가 세트 (2026-11-15). 기존 test 400 은 그대로 두고 따로 만든다.

  - unsupported: make_data.py 의 추가 유형 (unsup_object / unsup_event / unsup_race / unsup_check(2026-11-15)) × 학습 때 본 패턴·처음 보는 패턴(마지막 패턴)
    + combo: 두 유형을 합친 문장 (시연 문장 '화장실에서 관리자가 나오면 관리자보다 먼저 A1으로 가' 형태, 학습에 없는 구조)
    정답: {"plan": [], "error": "unsupported_command"}, 행동 거부
  - control: 새 데이터 때문에 정상 명령까지 거부하게 됐는지 보는 대조군. 정답 계획대로 실행
    · 비슷한 단어를 쓰는 정상 명령 (충전기·포장대·입고대로 이동, 충전)
    · '먼저 A. 그 후에 B' (경쟁 유형의 '먼저'), 'A. 끝나면 B' (사건 유형의 '~하면') 연결
    · control_mod (2026-11-15): 장소를 사람·물건으로 꾸미기만 하는 이동 ('관리자가 있는 B2로 가') × 본 패턴·처음 보는 패턴
      ('{사람}이 기다리는 {장소}' 패턴은 2026-11-16 부터 정답 go_to → dock)
    · control_inform (2026-11-16): 사람 위치 알림 ('관리자가 B2에서 기다리고 있어요, 가세요') → go_to → dock × 본 패턴·처음 보는 패턴
  - unknown (2026-11-16): 꾸밈말·위치 알림 + 목록에 없는 위치 ('직원이 있는 E3으로 가') → unknown_location 거부 × 본 패턴·처음 보는 패턴
  - 시연 문장 (demo): 할 수 없는 일 '충전기가져와', '화장실에서 관리자가 나오면 관리자보다 먼저 A1으로 가',
    '관리인이 A1에 있는지 가서 확인하고 없다면 D3으로 가' / 정상 이동 '관리인이 있는 A3으로 가' 등 3개 ('관리인' 은 학습에 없는 단어) /
    사람 위치 알림 '관리자가 A2에서 기다립니다 가세요' (정답 go_to → dock, 2026-11-16)
  - train/val/test 와 같은 문장은 뺀다 → make_data.py 를 먼저 실행한 뒤 실행할 것

실행: python make_probe_unsupported.py     결과: data/probe_unsupported.jsonl
각 줄: input, output(정답 JSON), template, heldout, kind(unsupported|unknown|control|demo), expected_action
"""
import json
import random

import make_data as md

N_PER = 20          # 유형 × (본 패턴 / 처음 보는 패턴) 마다
N_COMBO = 15
N_CONTROL = 10      # 대조군 종류마다
UNSUP = {"plan": [], "error": "unsupported_command"}
GO_A3 = {"plan": [{"skill": "go_to", "target": "A3"}]}
DEMO = [("충전기가져와", UNSUP, "거부"),
        ("화장실에서 관리자가 나오면 관리자보다 먼저 A1으로 가", UNSUP, "거부"),
        ("관리인이 A1에 있는지 가서 확인하고 없다면 D3으로 가", UNSUP, "거부"),        # 2026-11-15 시연
        ("관리인이 있는 A3으로 가", GO_A3, "실행"),
        ("지금 관리인이 있는 A3으로 가", GO_A3, "실행"),
        ("관리인이 없는 A3으로 가", GO_A3, "실행"),
        ("관리자가 A2에서 기다립니다 가세요", {"plan": [{"skill": "go_to", "target": "A2"},
                                                {"skill": "dock", "target": "A2"}]}, "실행")]   # 2026-11-15 시연, 정답 2026-11-16 변경


def main():
    random.seed(11)
    seen = set()
    for name in ["train", "val", "test"]:
        seen |= {json.loads(l)["input"] for l in open(f"data/{name}.jsonl", encoding="utf-8")}
    rows, used = [], set()

    def add(txt, out, template, heldout, kind, action):
        if txt in seen or txt in used:
            return False
        used.add(txt)
        rows.append({"input": txt, "output": json.dumps(out, ensure_ascii=False), "template": template,
                     "heldout": heldout, "kind": kind, "expected_action": action})
        return True

    def fill(n, make):
        got, tries = 0, 0
        while got < n and tries < n * 200:
            tries += 1
            got += make()

    # 할 수 없는 일 3유형 × 본 패턴·처음 보는 패턴
    for name, fn in md.UNSUP_EXTRA.items():
        for mode, held in [("train", False), ("heldout", True)]:
            pool = md.make_pool(mode)
            fill(N_PER, lambda: add(fn(pool)[0], UNSUP, name, held, "unsupported", "거부"))

    # 두 유형을 합친 문장 (학습에 없는 구조)
    def combo():
        who = random.choice(md.PEOPLE[:5])
        lp = md._any_loc()
        txt = f"{random.choice(md.ROOMS)}에서 {md.ga(who)} 나오면 {who}보다 먼저 {md.ro(lp)} 가"
        return add(txt, UNSUP, "unsup_combo", True, "unsupported", "거부")
    fill(N_COMBO, combo)

    # 대조군: 비슷한 단어를 쓰는 정상 명령
    def move_near():
        loc = random.choice(["charger", "packing", "inbound"])
        name = random.choice(md.SPECIAL_LOCATIONS[loc])     # 충전기·충전 스테이션·포장대·입고대 등 (NON_ITEMS 와 겹치는 이름)
        txt = random.choice([f"{md.ro(name)} 가", f"{name} 쪽으로 가줘", f"{name} 앞으로 가", f"{md.ro(name)} 이동해"])
        return add(txt, {"plan": md.p_go(loc)}, "control_move", False, "control", "실행")
    fill(N_CONTROL, move_near)

    def charge():
        txt, plan = md.t_charge(md.make_pool("train"))
        return add(txt, {"plan": plan}, "control_charge", False, "control", "실행")
    fill(N_CONTROL, charge)

    # 대조군: 장소를 사람·물건으로 꾸미기만 하는 이동 (2026-11-15) × 본 패턴·처음 보는 패턴
    for mode, held in [("train", False), ("heldout", True)]:
        def mod(pool=md.make_pool(mode), held=held):
            txt, plan = md.t_move_mod(pool)
            return add(txt, {"plan": plan}, "control_mod", held, "control", "실행")
        fill(N_PER, mod)

    # 대조군: '먼저 ~. 그 후에 ~' / '~. 끝나면 ~' 연결
    names = [n for n in md.SINGLE if n not in ("follow", "move_mod", "inform")]
    for conn, tag in [("먼저 {a}. 그 후에 {b}", "control_first"), ("{a}. 끝나면 {b}", "control_then")]:
        def chain(conn=conn, tag=tag):
            pool = md.make_pool("train")
            n1, n2 = random.sample(names, 2)
            a, pa = md.SINGLE[n1](pool)
            b, pb = md.SINGLE[n2](pool)
            txt = conn.format(a=md._ends_imperative(a), b=b)
            return add(txt, {"plan": pa + pb}, tag, False, "control", "실행")
        fill(N_CONTROL, chain)

    # 대조군: 사람 위치 알림 → go_to → dock / 거부: 꾸밈말·위치 알림 + 없는 위치 (2026-11-16) × 본 패턴·처음 보는 패턴
    for mode, held in [("train", False), ("heldout", True)]:
        def inform(pool=md.make_pool(mode), held=held):
            txt, plan = md.t_inform(pool)
            return add(txt, {"plan": plan}, "control_inform", held, "control", "실행")
        fill(N_PER, inform)
    for mode, held in [("train", False), ("heldout", True)]:
        def unknown(pool=md.make_pool(mode), held=held):
            txt, _, err = md.t_unknown_mod(pool)
            return add(txt, {"plan": [], "error": err}, "unknown_mod", held, "unknown", "거부")
        fill(N_PER, unknown)

    # 시연 문장 (학습 데이터에 같은 문장이 있으면 표시만 함)
    for txt, out, action in DEMO:
        rows.append({"input": txt, "output": json.dumps(out, ensure_ascii=False), "template": "demo",
                     "heldout": txt not in seen, "kind": "demo", "expected_action": action})

    with open("data/probe_unsupported.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    count = {}
    for r in rows:
        k = (r["template"], r["heldout"])
        count[k] = count.get(k, 0) + 1
    print(f"data/probe_unsupported.jsonl: {len(rows)}개")
    for (t, h), n in count.items():
        print(f"  {t:16s} {'처음 보는 패턴' if h else '본 패턴':8s} {n}")
    print("\n예시:")
    for r in random.sample(rows, 10):
        print(" ", r["input"], "→", r["expected_action"])


if __name__ == "__main__":
    main()
