"""장소 점수 모델(회귀): 명령 문장에서 '창고 업무 관계율' 과 '위험률' (0~1) 을 예측한다.

구조
  - 명령 해석 모델(파인튜닝한 LLM)은 그대로 두고(학습 안 함), 명령을 넣었을 때 나오는 내부 표현(hidden state)만 꺼낸다
  - 그 위에 작은 선형 회귀층(릿지 회귀)을 학습. 해석 모델이 한 번 읽은 결과를 재사용하므로 추가 비용이 거의 없음
  - 문장 표현 3가지 비교
      mean : 명령 문장 전체 토큰의 평균
      last : 마지막 토큰 (답을 쓰기 직전 위치)
      place: 장소 이름 토큰의 평균 (점수를 결정하는 건 장소라서, 다른 단어·답변 계획이 덜 섞임)
  - 관계율·위험률은 각자 가장 좋은 (방식, 층, 규제 세기) 를 따로 고른다

데이터
  - data/labels.csv 의 장소 × 문장 패턴 8개. 같은 장소의 문장은 정답 점수가 같음
  - 평가는 '장소 단위' 교차검증(5겹): 시험하는 장소의 문장은 학습에 하나도 안 들어감 → 처음 보는 장소에 대한 일반화 측정
    (각 겹에 장소 그룹이 고르게 들어가도록 나눔)
  - 장소 점수 = 그 장소 문장 8개 예측의 평균
  - 보고하는 최종 수치는 '중첩 교차검증': 바깥 겹의 시험 장소는 방식·층·alpha 선택에도 안 쓰임 → 낙관 편향 없음

실행 예:
  python train_scorer.py                                   # 기본: outputs/qwen35_2b_v2_strict/merged, strict 프롬프트
  python train_scorer.py --model unsloth/Qwen3.5-2B --name scorer_2b_base     # 파인튜닝 전 모델과 비교
  python train_scorer.py --model outputs/qwen35_08b_v2/merged --prompt default --name scorer_08b
  python train_scorer.py --extra data/labels_add.csv --name scorer_2b_strict_add   # 추가 장소는 학습에만, 평가는 같은 106곳

결과
  - 화면: 방식·층별 교차검증 오차(MAE)·순위 상관, 중첩 교차검증 결과, 위험 구역 지표, 행동 판단 시뮬레이션
  - results/<name>.csv : 장소별 정답 vs 중첩 교차검증 예측·행동 (튀는 장소 확인용)
  - results/<name>.json: 요약
  - outputs/scorer/<name>.pt : 전체 데이터로 다시 학습한 최종 회귀층 (outputs/ 는 git 에 안 올림)
  - outputs/scorer/<name>_feats.npz : 추출한 특징 캐시 (같은 모델·프롬프트·문장이면 다음 실행에서 재사용)
"""
import argparse
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch

from make_data import ro
from schema import PROMPTS

TARGETS = ["relevance", "risk"]
POOLS = ["mean", "last", "place"]

# 장소 하나로 문장 8개 (make_probe.py 와 같은 계열). '{' 앞은 고정 문자열이라 '{' 위치 = 문장 안 장소 시작 위치
PATTERNS = [
    "{pro} 가",
    "{pro} 이동해",
    "{p}에 가 있어",
    "택배 {p}에 내려놔",
    "{p}에서 상자 꺼내서 출고장에 갖다 놔",
    "A1에서 박스 꺼내서 {p}에 갖다 놔",
    "싣고 있는 거 {p}에 내려놔",
    "{p}까지 가 줄래?",
]


def render(pat, place):
    """(문장, 문장 안 장소 시작 위치)"""
    return pat.format(p=place, pro=ro(place)), pat.index("{")


# ---------------------------------------------------------------- 특징 추출
def token_span(offsets, cs, ce):
    """글자 구간 [cs, ce) 와 겹치는 토큰 구간 [start, end). 채움 토큰은 offset 이 (0,0) 이라 빠짐"""
    idx = [i for i, (a, b) in enumerate(offsets.tolist()) if b > a and a < ce and b > cs]
    return idx[0], idx[-1] + 1


@torch.no_grad()
def extract(model, tok, items, system, batch=32):
    """items: [(문장, 장소 시작 위치, 장소 글자 수)].
    반환: {방식: [문장 수, 층 수, 은닉 크기] float16}, 고른 층 번호 목록 (층 l = hidden_states[l+1], 0번은 임베딩)"""
    from evaluate import build_prompt
    feats, layers = {k: [] for k in POOLS}, None
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        prompts = [build_prompt(tok, t, system) for t, _, _ in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False,
                  return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping")
        hs = model(**enc.to("cuda"), output_hidden_states=True).hidden_states
        if layers is None:                                  # 2층마다 + 마지막 층
            n = len(hs) - 1
            layers = sorted(set(range(1, n, 2)) | {n - 1})
        hs = torch.stack([hs[l + 1] for l in layers], dim=1)          # [B, 층, 길이, 은닉]
        for b, (prompt, (t, ps, pl)) in enumerate(zip(prompts, chunk)):
            c0 = prompt.rfind(t)            # 시스템 프롬프트에도 같은 단어가 있을 수 있어서 마지막 위치(사용자 메시지)
            s, e = token_span(offsets[b], c0, c0 + len(t))
            ps_, pe_ = token_span(offsets[b], c0 + ps, c0 + ps + pl)
            feats["mean"].append(hs[b, :, s:e].float().mean(1).cpu())
            feats["place"].append(hs[b, :, ps_:pe_].float().mean(1).cpu())
            feats["last"].append(hs[b, :, -1].float().cpu())
        print(f"  특징 추출 {min(i + batch, len(items))}/{len(items)}", end="\r")
    print()
    return {k: torch.stack(v).half().numpy() for k, v in feats.items()}, layers


def load_or_extract(args, items):
    cache = f"outputs/scorer/{args.name}_feats.npz"
    texts = np.array([t for t, _, _ in items])
    if os.path.exists(cache):
        z = np.load(cache)
        if (str(z["model"]) == args.model and str(z["prompt"]) == args.prompt
                and len(z["texts"]) == len(texts) and (z["texts"] == texts).all()):
            print(f"특징 캐시 사용: {cache}")
            return {k: z[k] for k in POOLS}, z["layers"].tolist()
    from evaluate import load_model
    print(f"모델 불러오는 중: {args.model}")
    model, tok = load_model(args.model)
    t0 = time.time()
    feats, layers = extract(model, tok, items, PROMPTS[args.prompt])
    print(f"특징 추출 {time.time() - t0:.0f}초, 층 {layers}, 은닉 크기 {feats['mean'].shape[2]}")
    del model
    torch.cuda.empty_cache()
    os.makedirs("outputs/scorer", exist_ok=True)
    np.savez(cache, texts=texts, layers=np.array(layers), model=args.model, prompt=args.prompt, **feats)
    return feats, layers


# ---------------------------------------------------------------- 회귀
def ridge_fit(X, Y, alpha):
    """표준화 후 릿지 회귀 (표본 수 < 차원 이라 쌍대 형식으로 풂). Y: [n, 타깃 수]"""
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    ym = Y.mean(0)
    A = np.linalg.solve(Z @ Z.T + alpha * np.eye(len(Z)), Y - ym)
    return {"mu": mu, "sd": sd, "W": Z.T @ A, "b": ym}


def ridge_predict(m, X):
    return np.clip(((X - m["mu"]) / m["sd"]) @ m["W"] + m["b"], 0.0, 1.0)


def ridge_multi(Xtr, Ytr, Xte, alphas):
    """여러 alpha 의 시험 예측을 한 번에 (Z Zᵀ 고윳값 분해를 공유) → {alpha: [시험 수, 타깃 수]}"""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
    ym = Ytr.mean(0)
    lam, V = np.linalg.eigh(Ztr @ Ztr.T)
    VtY = V.T @ (Ytr - ym)
    KV = (Zte @ Ztr.T) @ V
    return {a: np.clip(KV @ (VtY / (lam + a)[:, None]) + ym, 0.0, 1.0) for a in alphas}


def make_folds(places, groups, k, seed=0):
    """장소 단위로 k겹 분할. 그룹마다 섞은 뒤 돌아가며 배정 → 각 겹에 그룹이 고르게"""
    rng = random.Random(seed)
    fold = {}
    for g in sorted(set(groups)):
        ps = [p for p, gg in zip(places, groups) if gg == g]
        rng.shuffle(ps)
        for j, p in enumerate(ps):
            fold[p] = j % k
    return fold


def spearman(a, b):
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))


def by_place(pred_rows, row_place):
    df = pd.DataFrame(pred_rows, columns=TARGETS)
    df["place"] = row_place
    return df.groupby("place")[TARGETS].mean()


def grid_cv(feats, Y, row_place, fold, alphas):
    """fold 에 있는 장소만 써서 모든 (방식, 층 위치, alpha) 의 장소 단위 교차검증 예측
    fold 값 -1 = 시험에는 안 쓰고 학습에만 쓰는 장소 (--extra)
    → {(방식, 층 위치, alpha): DataFrame(시험 장소 × 타깃)}"""
    rows = np.array([p in fold for p in row_place])
    rp = [p for p in row_place if p in fold]
    f_rows = np.array([fold[p] for p in rp])
    ev = f_rows >= 0
    rp_ev = [p for p, e in zip(rp, ev) if e]
    Yr = Y[rows]
    out = {}
    for pool in POOLS:
        for li in range(feats[pool].shape[1]):
            X = feats[pool][rows, li].astype(np.float64)
            pred = {a: np.zeros((len(rp), len(TARGETS))) for a in alphas}
            for f in sorted(set(f_rows[ev])):
                te = f_rows == f
                for a, pr in ridge_multi(X[~te], Yr[~te], X[te], alphas).items():
                    pred[a][te] = pr
            for a in alphas:
                out[(pool, li, a)] = by_place(pred[a][ev], rp_ev)
    return out


def best_key(grid, true_df, target):
    """target 의 장소 평균 절대 오차가 가장 작은 설정"""
    return min(grid, key=lambda k: (grid[k][target] - true_df.loc[grid[k].index, target]).abs().mean())


def nested_cv(feats, Y, row_place, places, extra, group_of, true_df, k, alphas):
    """바깥 겹: 시험 장소 / 안쪽 겹: 남은 장소로 (방식, 층, alpha) 선택 → 시험 장소는 선택에도 안 쓰임
    extra 장소는 언제나 학습 쪽에만 들어감"""
    outer = make_folds(places, [group_of[p] for p in places], k)
    pred = pd.DataFrame(index=places, columns=TARGETS, dtype=float)
    chosen = []
    for f in range(k):
        tr = [p for p in places if outer[p] != f]
        te = [p for p in places if outer[p] == f]
        inner = make_folds(tr, [group_of[p] for p in tr], k - 1, seed=f + 1)
        inner.update({p: -1 for p in extra})
        grid = grid_cv(feats, Y, row_place, inner, alphas)
        tr_set = set(tr) | set(extra)
        rows_tr = np.array([p in tr_set for p in row_place])
        rows_te = np.array([outer.get(p, -1) == f for p in row_place])
        for j, t in enumerate(TARGETS):
            pool, li, a = best_key(grid, true_df, t)
            X = feats[pool][:, li].astype(np.float64)
            pr = ridge_predict(ridge_fit(X[rows_tr], Y[rows_tr], a), X[rows_te])
            pred.loc[te, t] = by_place(pr, [p for p in row_place if outer.get(p, -1) == f]).loc[te, t]
            chosen.append({"fold": f, "target": t, "pool": pool, "layer_pos": li, "alpha": a})
        print(f"  중첩 교차검증 겹 {f + 1}/{k}", end="\r")
    print()
    return pred.loc[places].values, chosen


# ---------------------------------------------------------------- 행동 판단
def decide(rel, risk, exec_rel, ask_rel, risk_max):
    if rel >= exec_rel and risk < risk_max:
        return "실행"
    if rel >= ask_rel:
        return "확인"
    return "거부"


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="outputs/qwen35_2b_v2_strict/merged")
    ap.add_argument("--prompt", default="strict", choices=list(PROMPTS))
    ap.add_argument("--labels", default="data/labels.csv", help="평가(교차검증 시험)와 학습에 쓰는 장소")
    ap.add_argument("--extra", default="", help="학습에만 쓰는 장소 csv (쉼표로 여러 개). 평가 장소를 같게 두고 데이터 추가 효과를 볼 때")
    ap.add_argument("--name", default="scorer_2b_strict")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--alphas", default="1,10,100,1000,10000", help="릿지 규제 세기 후보")
    ap.add_argument("--exec_rel", type=float, default=0.9, help="실행: 관계율 이 값 이상 + 위험률 risk_max 미만")
    ap.add_argument("--ask_rel", type=float, default=0.5, help="확인 질문: 관계율 이 값 이상. 그 밖은 거부")
    ap.add_argument("--risk_max", type=float, default=0.5)
    ap.add_argument("--high_risk", type=float, default=0.8, help="위험 구역 지표 기준 (정답 위험률 이 값 이상)")
    args = ap.parse_args()
    alphas = [float(a) for a in args.alphas.split(",")]
    np.random.seed(0)

    lab = pd.read_csv(args.labels, encoding="utf-8-sig")
    ext = pd.concat([pd.read_csv(f, encoding="utf-8-sig") for f in args.extra.split(",") if f]) \
        if args.extra else lab.iloc[:0]
    places, extra = lab["place"].tolist(), ext["place"].tolist()
    assert not set(places) & set(extra), "평가 장소와 --extra 장소가 겹침"
    all_lab = pd.concat([lab, ext])
    group_of = dict(zip(all_lab["place"], all_lab["group"]))
    true_df = all_lab.set_index("place")[TARGETS].astype(np.float64)
    true = true_df.loc[places].values
    items, row_place = [], []
    for p in places + extra:
        for pat in PATTERNS:
            t, ps = render(pat, p)
            items.append((t, ps, len(p)))
            row_place.append(p)
    Y = true_df.loc[row_place].values
    print(f"평가 장소 {len(places)}곳 + 학습 전용 {len(extra)}곳, × 패턴 {len(PATTERNS)}개 = 문장 {len(items)}개")

    feats, layers = load_or_extract(args, items)

    # 기준선: 학습 장소들의 평균값만 예측
    fold = make_folds(places, lab["group"].tolist(), args.folds)
    fold.update({p: -1 for p in extra})
    base = np.zeros_like(true)
    for f in range(args.folds):
        te = np.array([fold[p] == f for p in places])
        base[te] = true_df.loc[[p for p in places + extra if fold[p] != f]].values.mean(0)
    bmae = np.abs(base - true).mean(0)
    print(f"\n기준선(평균값만 예측) MAE: 관계율 {bmae[0]:.3f} / 위험률 {bmae[1]:.3f}")

    # 1) 방식·층별 비교 (참고용. 같은 교차검증으로 고르므로 최솟값은 약간 낙관적)
    t0 = time.time()
    grid = grid_cv(feats, Y, row_place, fold, alphas)
    table = []
    print(f"\n{'방식':<6}{'층':>3} | {'관계율 MAE':>10}{'순위상관':>8} | {'위험률 MAE':>10}{'순위상관':>8}")
    for pool in POOLS:
        for li, l in enumerate(layers):
            row = {"pool": pool, "layer": l}
            for t in TARGETS:
                keys = [(pool, li, a) for a in alphas]
                k = best_key({kk: grid[kk] for kk in keys}, true_df, t)
                pr = grid[k].loc[places, t].values
                row[t] = {"alpha": k[2], "mae": float(np.abs(pr - true_df.loc[places, t]).mean()),
                          "rho": spearman(pr, true_df.loc[places, t].values)}
            table.append(row)
            r, k = row["relevance"], row["risk"]
            print(f"{pool:<6}{l:>3} | {r['mae']:>10.3f}{r['rho']:>8.3f} | {k['mae']:>10.3f}{k['rho']:>8.3f}")
    print(f"  ({time.time() - t0:.0f}초)")
    selected = {}
    for t in TARGETS:
        b = min(table, key=lambda r: r[t]["mae"])
        selected[t] = {"pool": b["pool"], "layer": b["layer"], **b[t]}
        print(f"  {t} 최적: {b['pool']} / 층 {b['layer']} / alpha {b[t]['alpha']:.0f} → MAE {b[t]['mae']:.3f}, 순위상관 {b[t]['rho']:.3f}")

    # 2) 중첩 교차검증 (정직한 일반화 수치)
    t0 = time.time()
    pred, chosen = nested_cv(feats, Y, row_place, places, extra, group_of, true_df, args.folds, alphas)
    mae = np.abs(pred - true).mean(0)
    rho = [spearman(pred[:, j], true[:, j]) for j in range(len(TARGETS))]
    print(f"중첩 교차검증 ({time.time() - t0:.0f}초) MAE: 관계율 {mae[0]:.3f} / 위험률 {mae[1]:.3f}"
          f"  순위상관: {rho[0]:.3f} / {rho[1]:.3f}  (기준선 MAE {bmae[0]:.3f} / {bmae[1]:.3f})")
    for t in TARGETS:
        cs = [f"{c['pool']}{layers[c['layer_pos']]}" for c in chosen if c["target"] == t]
        print(f"  겹별 선택 ({t}): {', '.join(cs)}")

    df = lab[["place", "group"] + TARGETS].copy()
    df["pred_relevance"], df["pred_risk"] = pred[:, 0].round(3), pred[:, 1].round(3)
    df["err_relevance"] = (df["pred_relevance"] - df["relevance"]).round(3)
    df["err_risk"] = (df["pred_risk"] - df["risk"]).round(3)

    # 3) 위험 구역 지표: 정답 위험률이 높은 곳을 얼마나 낮게 보는가 (안전에 중요한 쪽의 오류)
    hi = df[df["risk"] >= args.high_risk]
    hi_low = hi[hi["pred_risk"] < args.risk_max]
    slope = float(np.polyfit(df["risk"], df["pred_risk"], 1)[0])
    print(f"\n위험 구역 (정답 위험률 ≥ {args.high_risk}) {len(hi)}곳: 정답 평균 {hi['risk'].mean():.2f} → 예측 평균 "
          f"{hi['pred_risk'].mean():.2f}, 예측이 {args.risk_max} 미만인 곳 {len(hi_low)}곳"
          f"{' (' + ', '.join(hi_low['place']) + ')' if len(hi_low) else ''}")
    print(f"  위험률 예측 기울기(정답 1 오를 때 예측 증가량) {slope:.2f}  (1 에 가까울수록 좋음, 작으면 평균 쪽으로 눌림)")

    # 4) 행동 판단 시뮬레이션 (정답 점수로 정한 행동 vs 예측 점수로 정한 행동)
    th = (args.exec_rel, args.ask_rel, args.risk_max)
    df["action"] = [decide(r, k, *th) for r, k in zip(df["relevance"], df["risk"])]
    df["pred_action"] = [decide(r, k, *th) for r, k in zip(df["pred_relevance"], df["pred_risk"])]
    conf = pd.crosstab(df["action"], df["pred_action"]).reindex(
        index=["실행", "확인", "거부"], columns=["실행", "확인", "거부"], fill_value=0)
    print(f"\n행동 판단 (실행: 관계율 ≥ {th[0]} 이고 위험률 < {th[2]} / 확인: 관계율 ≥ {th[1]} / 그 밖 거부)")
    print("  행 = 정답 점수로 정한 행동, 열 = 예측 점수로 정한 행동")
    print("  " + conf.to_string().replace("\n", "\n  "))
    acc = float((df["action"] == df["pred_action"]).mean())
    wrong_exec = df[(df["pred_action"] == "실행") & (df["action"] != "실행")]
    wrong_reject = df[(df["action"] == "거부") & (df["pred_action"] != "거부")]
    print(f"  일치율 {acc:.1%}, 잘못 '실행' {len(wrong_exec)}곳"
          f"{' (' + ', '.join(wrong_exec['place']) + ')' if len(wrong_exec) else ''}, "
          f"거부할 곳을 거부 안 함 {len(wrong_reject)}곳")

    print("\n그룹별 평균 절대 오차 (관계율 / 위험률):")
    for g, sub in df.groupby("group"):
        print(f"  {g:<20} {sub['err_relevance'].abs().mean():.3f} / {sub['err_risk'].abs().mean():.3f}  ({len(sub)}곳)")
    worst = df.assign(e=df["err_risk"].abs()).nlargest(8, "e")
    print("\n위험률 오차가 큰 장소:")
    for _, r in worst.iterrows():
        print(f"  {r['place']:<12} 위험률 {r['risk']:.2f}→{r['pred_risk']:.2f}  관계율 {r['relevance']:.2f}→{r['pred_relevance']:.2f}")

    os.makedirs("results", exist_ok=True)
    df.to_csv(f"results/{args.name}.csv", index=False, encoding="utf-8-sig")

    # 최종 회귀층: 타깃마다 고른 설정으로 전체 데이터를 다시 학습해서 저장
    heads = {}
    for j, t in enumerate(TARGETS):
        s = selected[t]
        X = feats[s["pool"]][:, layers.index(s["layer"])].astype(np.float64)
        m = ridge_fit(X, Y[:, j:j + 1], s["alpha"])
        heads[t] = {"pool": s["pool"], "layer": s["layer"], "alpha": s["alpha"],
                    **{k: torch.tensor(v) for k, v in m.items()}}
    os.makedirs("outputs/scorer", exist_ok=True)
    torch.save({"model": args.model, "prompt": args.prompt, "targets": TARGETS, "heads": heads},
               f"outputs/scorer/{args.name}.pt")
    summary = {
        "model": args.model, "prompt": args.prompt, "labels": args.labels, "extra": args.extra,
        "n_places": len(places), "n_extra": len(extra), "baseline_mae": bmae.tolist(),
        "nested": {"mae": mae.tolist(), "rho": rho, "chosen": [{**c, "layer": layers[c["layer_pos"]]} for c in chosen]},
        "selected": selected,
        "high_risk": {"threshold": args.high_risk, "n": len(hi), "true_mean": float(hi["risk"].mean()),
                      "pred_mean": float(hi["pred_risk"].mean()), "pred_below_risk_max": hi_low["place"].tolist()},
        "risk_slope": slope,
        "decision": {"exec_rel": th[0], "ask_rel": th[1], "risk_max": th[2], "agreement": acc,
                     "confusion": conf.to_dict(), "wrong_exec": wrong_exec["place"].tolist(),
                     "wrong_reject": wrong_reject["place"].tolist()},
        "grid": table,
    }
    with open(f"results/{args.name}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(f"\n저장: results/{args.name}.csv, results/{args.name}.json, outputs/scorer/{args.name}.pt")


if __name__ == "__main__":
    main()
