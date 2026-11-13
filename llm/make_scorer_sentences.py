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

TASKS_SINGLE = {"move": 2, "fetch_deliver": 4, "fetch": 1.5, "deliver": 1.5, "charge": 1.0}   # 장소가 있는 작업
TASKS_ALL = list(md.SINGLE)                               # 연결에는 대기·따라가기도 씀
CHARGER_LABEL = (1.0, 0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--chain", type=float, default=0.55, help="두 작업 연결 문장 비율")
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--out", default="data/scorer_sentences.csv")
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
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    chain = df.template.str.contains(r"\+")
    print(f"{len(df)}문장 → {args.out} (두 작업 연결 {chain.sum()}, 모두 공식 위치 {df.key.str.startswith('#').sum()}, "
          f"목록 밖 장소 포함 {(~df.key.str.startswith('#')).sum()}곳 {df[~df.key.str.startswith('#')].key.nunique()}종)")
    print(df.sample(8, random_state=0)[["text", "relevance", "risk"]].to_string(index=False))


if __name__ == "__main__":
    main()
