"""행동 기준값 정하기: 예측 점수에 적용할 (실행 관계율, 확인 관계율, 실행 위험률 상한) 을
학습 전용 장소로 정하고, 평가용 106곳으로 확인한다.

목표 행동 = 라벨 구간 (train_scorer.LABEL_BANDS: 관계율 ≥ 0.9 이고 위험률 < 0.5 실행 / 관계율 ≥ 0.5 확인 / 그 밖 거부)
  - 라벨을 이 구간에 맞춰 매겼으므로(0.9 이상은 공식 위치만) 목표 행동은 고정. 예측 점수는 평균 쪽으로 눌려 있어서
    예측에 적용할 기준값은 라벨 구간과 다를 수 있음 → 그 기준값을 찾는다
판단 단위 = 문장. 실제로는 문장마다 판단하므로 장소 평균이 아닌 문장별 예측을 쓴다
데이터 역할 (2026-11-13 결정과 같음: 평가용 106곳은 고르는 데 쓰지 않음)
  - 기준값 고르기: 학습 전용 장소(labels_add·labels_alias)의 문장. 예측은 그 장소를 빼고(106곳 + 나머지 학습 전용 장소로)
    학습한 점수 모델의 것 (중첩 교차검증)
  - 확인: 평가용 106곳의 문장. 예측은 train_scorer.py 와 같은 중첩 교차검증
채택 규칙 (2026-11-13 확정) — 임의의 비용 가중치 없이 기준값마다 역할을 나눔
  - 실행 관계율 0.9 · 확인 관계율 0.5: 라벨 구간의 의미 그대로 (0.9 이상은 공식 위치, 0.5 미만은 무관·위험)
  - 실행 위험률 상한: 학습 전용 공식 위치 문장의 예측 위험률 95% 백분위를 0.05 단위로 올림
    → 위험률 관문이 공식 위치 명령을 5% 넘게 막지 않게 하고, 나머지 걸러내기는 관계율이 맡음
참고로 함께 출력하는 '비용 최소 탐색' (채택 안 함: 가중치가 임의라 결과가 가중치에 좌우되고 지나치게 보수적)
            실행 확인 거부      (행 = 목표 행동, 열 = 예측 행동)
    실행      0    1    2
    확인      5    0    1
    거부     10    1    0     + 정답 위험률 0.5 이상인 장소를 '실행' 하면 50

실행: python tune_thresholds.py                 (문장별 예측을 다시 계산, 오래 걸림)
      python tune_thresholds.py --reuse_rows    (저장된 results/<name>_rows.csv 로 기준값만 다시 정함)
결과: results/thresholds.json (기준값·혼동표), results/<name>_rows.csv (문장별 중첩 교차검증 예측, e2e_check.py 가 사용)
"""
import argparse
import itertools
import json
import os

import numpy as np
import pandas as pd

from train_scorer import LABEL_BANDS, build_dataset, decide, load_or_extract, nested_cv

ACTS = ["실행", "확인", "거부"]
COST = np.array([[0, 1, 2], [5, 0, 1], [10, 1, 0]])
DANGER = 50
GRID = {"exec_rel": np.round(np.arange(0.70, 0.981, 0.01), 2),
        "ask_rel": np.round(np.arange(0.20, 0.701, 0.02), 2),
        "risk_max": np.round(np.arange(0.35, 0.751, 0.05), 2)}


def codes(rel, risk, e, a, r):
    """0 실행 / 1 확인 / 2 거부 (train_scorer.decide 와 같은 규칙, 벡터 계산)"""
    return np.where((rel >= e) & (risk < r), 0, np.where(rel >= a, 1, 2))


def total_cost(pred, tgt, true_risk, th):
    pa = codes(pred[:, 0], pred[:, 1], *th)
    danger = int(((pa == 0) & (true_risk >= 0.5)).sum())
    return int(COST[tgt, pa].sum()) + DANGER * danger, danger, pa


def confusion(tgt, pa):
    m = pd.crosstab(pd.Series([ACTS[i] for i in tgt], name="목표"), pd.Series([ACTS[i] for i in pa], name="예측"))
    return m.reindex(index=ACTS, columns=ACTS, fill_value=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="outputs/qwen35_2b_v2_strict/merged")
    ap.add_argument("--prompt", default="strict")
    ap.add_argument("--labels", default="data/labels.csv")
    ap.add_argument("--extra", default="data/labels_add.csv,data/labels_alias.csv")
    ap.add_argument("--sentences", default="data/scorer_sentences.csv")
    ap.add_argument("--name", default="scorer_2b_strict_sent", help="특징 캐시 이름 (train_scorer.py 의 --name)")
    ap.add_argument("--method", default="ensemble", choices=["select", "ensemble"])
    ap.add_argument("--ens_pool", default="last")
    ap.add_argument("--ens_layers", default="9,11,13,15,17,19,21")
    ap.add_argument("--alphas", default="1,10,100,1000,10000")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3, help="장소 분할을 바꿔 반복한 예측의 평균을 씀")
    ap.add_argument("--reuse_rows", action="store_true", help="저장된 문장별 예측으로 기준값만 다시 정함")
    ap.add_argument("--risk_pct", type=float, default=95, help="위험률 상한 = 공식 위치 문장 예측 위험률의 이 백분위")
    ap.add_argument("--out", default="results/thresholds.json", help="기준값 결과 파일 (다른 해석 모델이면 다른 이름으로)")
    args = ap.parse_args()
    rows_path = f"results/{args.name}_rows.csv"
    if args.reuse_rows and os.path.exists(rows_path):
        print(f"저장된 문장별 예측 사용: {rows_path}")
        df = pd.read_csv(rows_path, encoding="utf-8-sig")
    else:
        df = heldout_rows(args)
        df.to_csv(rows_path, index=False, encoding="utf-8-sig")
    select_and_report(df, args, rows_path)


def heldout_rows(args):
    alphas = [float(a) for a in args.alphas.split(",")]
    ds = build_dataset(args.labels, args.extra, args.sentences)
    places, extra, true_df, group_of = ds["places"], ds["extra"], ds["true_df"], ds["group_of"]
    items, row_place, Y, trainable = ds["items"], ds["row_place"], ds["Y"], ds["trainable"]
    feats, layers = load_or_extract(args, items)
    ens_configs = [(args.ens_pool, layers.index(int(l))) for l in args.ens_layers.split(",")]

    # 문장별 중첩 교차검증 예측: 평가 장소(학습 전용은 늘 학습 쪽) / 학습 전용 장소(평가 장소는 늘 학습 쪽)
    rows = np.full((len(row_place), 2), np.nan)
    for split, (te_places, tr_always) in {"eval": (places, extra), "extra": (extra, places)}.items():
        acc = []
        for r in range(args.repeats):
            _, _, rp = nested_cv(feats, Y, row_place, te_places, tr_always, group_of, true_df, args.folds, alphas,
                                 args.method, ens_configs, seed=r, trainable=trainable)
            acc.append(rp)
            print(f"  {split} 중첩 교차검증 {r + 1}/{args.repeats}", end="\r")
        m = np.array([p in set(te_places) for p in row_place])
        rows[m] = np.mean(acc, 0)[m]
    print()
    df = pd.DataFrame({"text": [t for t, _, _ in items], "place": row_place,
                       "group": [group_of[p] for p in row_place],
                       "split": ["eval" if p in set(places) else "extra" for p in row_place],
                       "kind": ds["kind"],
                       "trainable": trainable, "relevance": Y[:, 0], "risk": Y[:, 1],
                       "pred_relevance": rows[:, 0].round(4), "pred_risk": rows[:, 1].round(4)})
    df["target"] = [ACTS.index(decide(r, k, *LABEL_BANDS)) for r, k in zip(df.relevance, df.risk)]
    return df


def select_and_report(df, args, rows_path):
    tune, ev = df[df.split == "extra"], df[df.split == "eval"]
    P = lambda d: d[["pred_relevance", "pred_risk"]].values

    # 채택 규칙: 관계율은 라벨 구간, 위험률 상한은 학습 전용 공식 위치 문장 예측 위험률의 백분위 (0.05 단위 올림)
    p = float(np.percentile(tune[tune.target == 0].pred_risk, args.risk_pct))
    adopted = (LABEL_BANDS[0], LABEL_BANDS[1], float(np.round(np.ceil(p / 0.05 - 1e-9) * 0.05, 2)))
    print(f"\n채택 기준값: 실행 관계율 ≥ {adopted[0]}, 확인 관계율 ≥ {adopted[1]}, 실행 위험률 < {adopted[2]}"
          f"  (학습 전용 공식 위치 문장 예측 위험률 {args.risk_pct:g}% 백분위 {p:.3f})")

    # 참고: 비용 최소 탐색 (학습 전용 장소 문장)
    res = []
    for th in itertools.product(GRID["exec_rel"], GRID["ask_rel"], GRID["risk_max"]):
        if th[1] >= th[0]:
            continue
        c, dg, _ = total_cost(P(tune), tune.target.values, tune.risk.values, th)
        res.append((c, sum(abs(x - y) for x, y in zip(th, LABEL_BANDS)), th, dg))
    best_cost = min(r[0] for r in res)
    best = min([r for r in res if r[0] == best_cost], key=lambda r: r[1])
    th = tuple(float(x) for x in best[2])
    same = [r[2] for r in res if r[0] == best_cost and r[2][1:] == best[2][1:]]
    e_range = (float(min(s[0] for s in same)), float(max(s[0] for s in same)))
    print(f"참고 — 비용 최소 탐색 (채택 안 함): {th}  (비용 {best_cost})")

    summary = {"thresholds": {"exec_rel": adopted[0], "ask_rel": adopted[1], "risk_max": adopted[2]},
               "rule": f"관계율 = 라벨 구간, 위험률 상한 = 학습 전용 공식 위치 문장 예측 위험률 {args.risk_pct:g}% 백분위({p:.3f})의 0.05 단위 올림",
               "label_bands": LABEL_BANDS, "cost_search": {"thresholds": th, "cost": best_cost, "cost_matrix": COST.tolist(),
                                                           "danger_cost": DANGER, "exec_rel_plateau": e_range},
               "rows": rows_path, "sets": {}}
    for name, d in [("tune", tune), ("eval", ev)]:
        for tag, t_ in [("adopted", adopted), ("cost_search", th), ("label_bands", LABEL_BANDS)]:
            c, dg, pa = total_cost(P(d), d.target.values, d.risk.values, t_)
            conf = confusion(d.target.values, pa)
            acc = float((pa == d.target.values).mean())
            wrong = d[(pa == 0) & (d.target.values != 0)]
            summary["sets"][f"{name}_{tag}"] = {
                "cost": c, "danger_exec": dg, "agreement": acc, "confusion": conf.to_dict(),
                "wrong_exec": wrong[["text", "place", "relevance", "risk", "pred_relevance", "pred_risk"]].to_dict("records")}
            label = "학습 전용(고르기)" if name == "tune" else "평가 106곳(확인)"
            off = float((pa[d.target.values == 0] == 0).mean())
            print(f"\n[{label}] {dict(adopted='채택', cost_search='비용 최소', label_bands='라벨 구간')[tag]} {t_}: "
                  f"문장 일치 {acc:.1%}, 공식 위치 실행 {off:.1%}, 잘못 실행 {len(wrong)}문장 (위험률 0.5 이상 {dg})")
            summary["sets"][f"{name}_{tag}"]["official_exec"] = off
            print("  " + conf.to_string().replace("\n", "\n  "))
            if name == "eval" and len(wrong):
                for p, n in wrong.place.value_counts().items():
                    print(f"    잘못 실행: {p} ({n}/8 문장)")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(f"\n저장: {args.out} ({rows_path} 기준)")


if __name__ == "__main__":
    main()
