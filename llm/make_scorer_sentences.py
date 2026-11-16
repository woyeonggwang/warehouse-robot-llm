"""점수 모델 학습 문장 다양화: make_data.py 의 학습용 문장 패턴으로 실제 명령 형태의 문장(한 작업·두 작업 연결)을 만든다.

배경: 연결 점검(e2e_check.py)에서 점수 모델이 장소 하나짜리 패턴 8개로만 학습해서, 공식 위치만 있는 정상 명령도
      여러 작업이 이어지면 점수가 떨어져 막힘 (두 작업 연결 정상 명령 134개 중 53개만 실행)

지키는 것 (2026-11-13 사용자 결정)
  - 문장 패턴: make_data.py 의 학습용만 (make_pool('train') = 각 템플릿의 마지막 패턴 제외, CONNECTORS = '다 하면' 제외)
    → test.jsonl 의 처음 보는 패턴이 섞이지 않아 해석 모델과 같은 기준으로 비교 가능
  - 장소: 평가용 106곳(data/labels.csv)은 쓰지 않음 → '처음 보는 장소' 평가 유지
      공식 위치 자리 = labels_alias.csv 의 공식 위치 다른 이름 42개 (선반 자리는 선반 별칭, 입고·출고·포장·충전 자리는 같은 위치의 별칭)
      목록 밖 장소 = labels_add.csv 75곳 + labels_alias.csv 경계 장소 16곳 (학습 전용)
  - test.jsonl 과 똑같은 문장은 뺌
정답 점수 (2026-11-13 결정): 문장 속 장소 중 가장 낮은 관계율 · 가장 높은 위험률
  ('충전하러 가' 처럼 장소 이름 없이 충전소로 가는 작업은 충전소 1.0/0.3 으로 봄)
문장 구성: 절반은 모든 자리가 공식 위치, 절반은 한 자리만 목록 밖 장소 (같은 문장 구조에서 장소 하나가 점수를 정함을 배우게)

결과: data/scorer_sentences.csv
  - key: 목록 밖 장소가 있으면 그 장소 이름 (교차검증에서 그 장소의 다른 문장과 같은 묶음), 모두 공식 위치면 '#공식N'
  - place_start, place_len: 목록 밖 장소의 문장 안 위치 (모두 공식 위치면 문장 전체). train_scorer 의 place 표현용

실행: python make_scorer_sentences.py --n 1500
"""
import argparse
import json
import random

import pandas as pd

import make_data as md

# 사람·물건이 나오는 문장 (2026-11-15, --person N). 2B(v4)는 다시 학습하지 않고 점수 모델만 보강
#   v4 점수 모델이 꾸밈말 이동('직원이 있는 B2로 가')·사람 위치 알림('~가 ~에서 기다립니다 가세요')을 관계율 낮게·위험률 높게 봐서
#   2B 가 맞게 해석해도 '확인' 으로 보냄 → 정상 명령(장소 점수 그대로)과 대조용 사건 대기(거부)를 넣음
#   평가 전용으로 남김: 시연 '관리자가 A2에서 기다립니다 가세요' 형태, '관리인'(make_data.PEOPLE 에 없음),
#   '{사람}이 작업 중인 {장소}로 가 볼래?'(t_move_mod 마지막 패턴, make_pool('train') 이 뺌)
INFORM = [   # 사람이 있는 곳을 알려 주며 가라는 정상 명령 → 장소 점수 그대로
    lambda p, lp: f"{md.ga(p)} {lp}에서 기다리고 있어요, 가세요",
    lambda p, lp: f"{md.ga(p)} {lp}에 있으니 거기로 가",
    lambda p, lp: f"{lp}에서 {md.ga(p)} 기다리고 있어. 그쪽으로 가",
    lambda p, lp: f"{md.ga(p)} {lp}에 있어요. 가 주세요",
    lambda p, lp: f"{md.ga(p)} {lp}에서 기다리니까 거기로 이동해",
    lambda p, lp: f"{lp}에 {md.ga(p)} 와 있으니 가 봐",
]
EVENT = [    # 사건 대기: 로봇은 누가 오거나 도착하는 것을 알 수 없음 (그럴 계획도 없음, 2026-11-15 사용자) → 거부
    lambda w, lp: f"{md.ga(w)} 오면 {md.ro(lp)} 가",
    lambda w, lp: f"{md.ga(w)} {lp}에 도착할 때까지 기다렸다가 가",
    lambda w, lp: f"{md.ga(w)} 도착할 때까지 기다렸다가 {md.ro(lp)} 가",
    lambda w, lp: f"{md.ga(w)} 부르면 {md.ro(lp)} 이동해",
    lambda w, lp: f"{lp}에서 {md.ga(w)} 올 때까지 기다렸다가 가",
    lambda w, lp: f"{md.ga(w)} {lp}에 오면 거기로 가",
]
EVENT_WHO = md.PEOPLE + ["트럭", "택배 차"]
EVENT_REL = 0.3      # 사건 대기의 정답 관계율 (거부 구간 < 0.5). 위험률은 장소 점수 그대로
TASKS_SINGLE = {"move": 2, "fetch_deliver": 4, "fetch": 1.5, "deliver": 1.5, "charge": 1.0}   # 장소가 있는 작업
TASKS_ALL = [n for n in md.SINGLE if n not in ("move_mod", "inform")]   # 연결에는 대기·따라가기도 씀 (move_mod·inform 은
                                    # 2026-11-15·16 추가된 해석 모델 데이터라 제외 → 다시 실행해도 같은 data/scorer_sentences.csv)
CHARGER_LABEL = (1.0, 0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--chain", type=float, default=0.55, help="두 작업 연결 문장 비율")
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--out", default="data/scorer_sentences.csv")
    ap.add_argument("--person", type=int, default=0,
                    help="기존 문장 뒤에 사람·물건 문장을 이만큼 덧붙임 (꾸밈말·위치 알림·사건 대기 1/3씩). 0 이면 기존과 같은 파일")
    args = ap.parse_args()
    random.seed(args.seed)
    rng = random.Random(args.seed + 1)            # 목록 밖 장소 고르기용 (문장 구조용 random 과 따로 둬서 구조가 같게)

    alias = pd.read_csv("data/labels_alias.csv", encoding="utf-8-sig")
    add = pd.read_csv("data/labels_add.csv", encoding="utf-8-sig")
    evaluation = set(pd.read_csv("data/labels.csv", encoding="utf-8-sig").place)
    off = alias[alias.group == "official"]
    non = pd.concat([add, alias[alias.group != "official"]])
    assert not (set(off.place) | set(non.place)) & evaluation, "평가용 장소가 섞임"
    label = {r.place: (r.relevance, r.risk) for r in pd.concat([off, non]).itertuples()}
    by_loc = {loc: off[off.nearest == loc].place.tolist() for loc in md.SPECIAL_LOCATIONS}
    shelf_alias = off[off.nearest.isin(md.SHELVES)].place.tolist()
    non_places = non.place.tolist()
    test_inputs = {json.loads(l)["input"] for l in open("data/test.jsonl", encoding="utf-8")}

    # make_data 의 장소 표현 함수를 바꿔 끼움: 자리 종류에 맞는 공식 별칭을 쓰고, 정한 자리 하나는 목록 밖 장소로
    state = {"slots": [], "bad_at": None, "bad": None}

    def loc_phrase(loc):
        ph = random.choice(by_loc[loc] if loc in md.SPECIAL_LOCATIONS else shelf_alias)
        if len(state["slots"]) == state["bad_at"]:
            ph = state["bad"]
        state["slots"].append(ph)
        return ph

    md.loc_phrase = loc_phrase
    md.SPECIAL_LOCATIONS["charger"] = by_loc["charger"]   # t_charge 의 '{충전소}로 가서 충전해' 패턴도 별칭으로 (이 프로세스 안에서만)
    pool = md.make_pool("train")
    names = list(TASKS_SINGLE)

    def one():
        """(문장, 템플릿 이름, 작업 이름들). 장소 기록은 state['slots']"""
        state["slots"] = []
        if random.random() < args.chain:
            w = [md.SINGLE_WEIGHTS[n] for n in TASKS_ALL]
            n1, n2 = random.choices(TASKS_ALL, w, k=2)
            while n1 in md.NO_FIRST or n1 == n2:
                n1, n2 = random.choices(TASKS_ALL, w, k=2)
            a, _ = md.SINGLE[n1](pool)
            b, _ = md.SINGLE[n2](pool)
            return random.choice(md.CONNECTORS).format(a=md._ends_imperative(a), b=b), f"{n1}+{n2}", (n1, n2)
        n = random.choices(names, [TASKS_SINGLE[x] for x in names])[0]
        txt, _ = md.SINGLE[n](pool)
        return txt, n, (n,)

    rows, seen, k_off, tries = [], set(), 0, 0
    while len(rows) < args.n and tries < args.n * 20:
        tries += 1
        s0 = random.getstate()
        state["bad_at"] = None
        txt, tmpl, tasks = one()
        n_slots = len(state["slots"])
        if n_slots == 0 and "charge" not in tasks:
            continue                                       # 장소 없는 문장 (대기+따라가기 등) 은 점수 모델을 안 거침
        bad = None
        if n_slots and rng.random() < 0.5:                 # 같은 구조로 한 자리만 목록 밖 장소로 바꿔 다시 만듦
            bad = rng.choice(non_places)
            random.setstate(s0)
            state["bad_at"], state["bad"] = rng.randrange(n_slots), bad
            txt, tmpl, tasks = one()
        if txt in seen or txt in test_inputs:
            continue
        seen.add(txt)
        scores = [label[p] for p in state["slots"]] + ([CHARGER_LABEL] if "charge" in tasks else [])
        rel, risk = min(s[0] for s in scores), max(s[1] for s in scores)
        if bad:
            assert (rel, risk) == label[bad]
            key, group, start = bad, None, txt.find(bad)
            span = (start, len(bad))
        else:
            key, group, span = f"#공식{k_off}", "official_sentence", (0, len(txt))
            k_off += 1
        rows.append({"text": txt, "key": key, "relevance": rel, "risk": risk, "group": group,
                     "place_start": span[0], "place_len": span[1], "template": tmpl,
                     "places": "|".join(state["slots"])})
    if args.person:
        rows += person_rows(args, state, label, non_places, seen | test_inputs)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    if args.person:
        for t, g in df[df.template.str.startswith("person_")].groupby("template"):
            print(f"  {t}: {len(g)}문장 (목록 밖 장소 포함 {(~g.key.str.startswith('#')).sum()}), "
                  f"정답 관계율 평균 {g.relevance.mean():.2f} · 위험률 평균 {g.risk.mean():.2f}")
    chain = df.template.str.contains(r"\+")
    print(f"{len(df)}문장 → {args.out} (두 작업 연결 {chain.sum()}, 모두 공식 위치 {df.key.str.startswith('#').sum()}, "
          f"목록 밖 장소 포함 {(~df.key.str.startswith('#')).sum()}곳 {df[~df.key.str.startswith('#')].key.nunique()}종)")
    print(df.sample(8, random_state=0)[["text", "relevance", "risk"]].to_string(index=False))


def person_rows(args, state, label, non_places, taken):
    """사람·물건 문장. 꾸밈말·위치 알림은 절반을 목록 밖 장소로 바꿔 만들고(정답 = 그 장소 점수),
    사건 대기는 공식 위치만 (정답 관계율 EVENT_REL, 위험률 = 장소 위험률, 교차검증 묶음은 '#사건N')"""
    probe = {json.loads(l)["input"] for l in open("data/probe_unsupported.jsonl", encoding="utf-8")}
    taken = taken | probe                                  # 추가 평가 세트(시연 문장 포함) 와 같은 문장은 뺌
    random.seed(args.seed + 100)
    rng = random.Random(args.seed + 101)
    pool = md.make_pool("train")
    locs = md.SHELVES + list(md.SPECIAL_LOCATIONS)

    def gen(kind):
        state["slots"] = []
        if kind == "mod":
            return md.t_move_mod(pool)[0]
        loc = random.choice(locs)
        if kind == "inform":
            p, f = random.choice(md.PEOPLE), random.choice(INFORM)
            return f(p, md.loc_phrase(loc))
        w, f = random.choice(EVENT_WHO), random.choice(EVENT)
        return f(w, md.loc_phrase(loc))

    out, made, k = [], {"mod": 0, "inform": 0, "event": 0}, {"mod": 0, "inform": 0, "event": 0}
    n_each, tries = args.person // 3, 0
    while min(made.values()) < n_each and tries < args.person * 50:
        tries += 1
        kind = min(made, key=made.get)
        s0 = random.getstate()
        state["bad_at"] = None
        txt = gen(kind)
        bad = None
        if kind != "event" and rng.random() < 0.5:
            bad = rng.choice(non_places)
            random.setstate(s0)
            state["bad_at"], state["bad"] = 0, bad
            txt = gen(kind)
        if txt in taken:
            continue
        taken.add(txt)
        rel, risk = label[state["slots"][0]]
        if kind == "event":
            rel, key, group = EVENT_REL, f"#사건{k[kind]}", "unsupported_sentence"
        elif bad:
            key, group = bad, None
        else:
            key, group = f"#{kind}{k[kind]}", "official_sentence"
        span = (txt.find(bad), len(bad)) if bad else (0, len(txt))
        k[kind] += 1
        made[kind] += 1
        out.append({"text": txt, "key": key, "relevance": rel, "risk": risk, "group": group,
                    "place_start": span[0], "place_len": span[1], "template": f"person_{kind}",
                    "places": "|".join(state["slots"])})
    return out


if __name__ == "__main__":
    main()
