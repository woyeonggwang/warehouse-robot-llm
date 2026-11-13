"""장소 점수 모델(회귀): 명령 문장에서 '창고 업무 관계율' 과 '위험률' (0~1) 을 예측한다.

구조
  - 명령 해석 모델(파인튜닝한 LLM)은 그대로 두고(학습 안 함), 명령을 넣었을 때 나오는 내부 표현(hidden state)만 꺼낸다
  - 그 위에 작은 선형 회귀층(릿지 회귀)을 학습. 해석 모델이 한 번 읽은 결과를 재사용하므로 추가 비용이 거의 없음
  - 문장 표현 3가지
      mean : 명령 문장 전체 토큰의 평균
      last : 마지막 토큰 (답을 쓰기 직전 위치)
      place: 장소 이름 토큰의 평균
  - 회귀층을 정하는 방식 (--method)
      ensemble (기본): 정해 둔 층들(last 9~21층, 7개)에 각각 회귀층을 학습하고 예측을 평균. 고르는 것은 alpha 하나뿐
      select  : (표현, 층, alpha) 180개 후보 중 교차검증 오차가 가장 작은 하나를 타깃마다 고름 (예전 방식.
                데이터·분할이 조금만 바뀌어도 고르는 층이 바뀌어 장소별 예측이 최대 0.56 흔들림)

데이터
  - data/labels.csv 의 장소 × 문장 패턴 8개. 같은 장소의 문장은 정답 점수가 같음
  - 평가는 '장소 단위' 교차검증(5겹): 시험하는 장소의 문장은 학습에 하나도 안 들어감 → 처음 보는 장소에 대한 일반화 측정
    (각 겹에 장소 그룹이 고르게 들어가도록 나눔)
  - 장소 점수 = 그 장소 문장 8개 예측의 평균
  - 보고하는 최종 수치는 '중첩 교차검증': 바깥 겹의 시험 장소는 설정 선택에도 안 쓰임 → 낙관 편향 없음
  - --repeats N: 장소 분할만 바꿔 중첩 교차검증을 N번 반복 → 장소별 예측이 분할에 따라 얼마나 흔들리는지 측정

  - test 전용 형태 패턴(HELDOUT_PATTERNS)은 예측(평가)만 하고 학습에는 안 씀
  - --sentences: make_scorer_sentences.py 가 만든 실제 명령 형태 문장 (학습 전용 장소·공식 별칭만, 늘 학습 쪽)

실행 예:
  python train_scorer.py --extra data/labels_add.csv,data/labels_alias.csv --sentences data/scorer_sentences.csv --name scorer_2b_strict_sent --repeats 3
  python train_scorer.py --extra data/labels_add.csv,data/labels_alias.csv --name scorer_2b_strict_add_alias
  python train_scorer.py --method select --name scorer_2b_strict                 # 예전 방식, 106곳만
  python train_scorer.py --model unsloth/Qwen3.5-2B --name scorer_2b_base         # 파인튜닝 전 모델과 비교

결과
  - 화면: 방식·층별 교차검증 오차(MAE)·순위 상관, 중첩 교차검증 결과, 위험 구역 지표, 행동 판단 시뮬레이션
  - results/<name>.csv : 장소별 정답 vs 중첩 교차검증 예측(반복 평균)·행동
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
# 위 패턴 중 make_data.py 의 test 전용(각 템플릿 마지막) 패턴과 같은 형태 → 예측(평가)만 하고 학습에는 안 씀 (2026-11-13)
#   "{p}까지 가 줄래?" = 이동(t_move) 마지막 패턴, "택배 {p}에 내려놔" = 없는 위치 거부(t_unknown_location) 마지막 패턴
HELDOUT_PATTERNS = {"{p}까지 가 줄래?", "택배 {p}에 내려놔"}

# 라벨 구간으로 정한 목표 행동 (채점 기준: 0.9 이상은 공식 위치만, 0.5~0.85 목록 밖 물류, 0.5 미만 무관·위험)
LABEL_BANDS = (0.9, 0.5, 0.5)


def render(pat, place):
    """(문장, 문장 안 장소 시작 위치)"""
    return pat.format(p=place, pro=ro(place)), pat.index("{")


# ---------------------------------------------------------------- 특징 추출
def token_span(offsets, cs, ce):
    """글자 구간 [cs, ce) 와 겹치는 토큰 구간 [start, end). 채움 토큰은 offset 이 (0,0) 이라 빠짐"""
    idx = [i for i, (a, b) in enumerate(offsets.tolist()) if b > a and a < ce and b > cs]
    return idx[0], idx[-1] + 1


@torch.no_grad()
def extract(model, tok, items, system, batch=16, verbose=True):
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
        # logits_to_keep=1: 다음 토큰 확률은 마지막 위치만 계산 (전체 위치를 계산하면 어휘 약 25만 개라
        # 32문장 × 252토큰에서 logits 만 4GB → GPU 메모리 부족, 2026-11-13). 내부 표현(hidden_states)은 그대로 전부 나옴
        hs = model(**enc.to("cuda"), output_hidden_states=True, logits_to_keep=1).hidden_states
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
        if verbose:
            print(f"  특징 추출 {min(i + batch, len(items))}/{len(items)}", end="\r")
    if verbose:
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
    """표준화 후 릿지 회귀. 표본 수 ≤ 차원이면 쌍대(Z Zᵀ), 표본이 더 많으면 원래 형식(Zᵀ Z)으로 풀어 계산량을 줄임. Y: [n, 타깃 수]"""
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    ym = Y.mean(0)
    if len(Z) <= Z.shape[1]:
        W = Z.T @ np.linalg.solve(Z @ Z.T + alpha * np.eye(len(Z)), Y - ym)
    else:
        W = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (Y - ym))
    return {"mu": mu, "sd": sd, "W": W, "b": ym}


def ridge_predict(m, X):
    return np.clip(((X - m["mu"]) / m["sd"]) @ m["W"] + m["b"], 0.0, 1.0)


def ridge_multi(Xtr, Ytr, Xte, alphas):
    """여러 alpha 의 시험 예측을 한 번에 (Z Zᵀ 고윳값 분해를 공유) → {alpha: [시험 수, 타깃 수]}"""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
    ym = Ytr.mean(0)
    if len(Ztr) <= Ztr.shape[1]:                    # 쌍대 형식
        lam, V = np.linalg.eigh(Ztr @ Ztr.T)
        VtY = V.T @ (Ytr - ym)
        KV = (Zte @ Ztr.T) @ V
        return {a: np.clip(KV @ (VtY / (lam + a)[:, None]) + ym, 0.0, 1.0) for a in alphas}
    lam, Q = np.linalg.eigh(Ztr.T @ Ztr)            # 원래 형식 (표본이 차원보다 많을 때)
    QtZtY = Q.T @ (Ztr.T @ (Ytr - ym))
    ZQ = Zte @ Q
    return {a: np.clip(ZQ @ (QtZtY / (lam + a)[:, None]) + ym, 0.0, 1.0) for a in alphas}


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


def grid_cv(feats, Y, row_place, fold, alphas, configs=None, trainable=None):
    """fold 에 있는 장소만 써서 (방식, 층 위치) 설정·alpha 별 장소 단위 교차검증 예측
    fold 값 -1 = 시험에는 안 쓰고 학습에만 쓰는 장소 (--extra). configs 가 없으면 모든 (방식, 층 위치)
    trainable: 문장별 학습 사용 여부 (False 인 문장은 예측만 하고 학습에는 안 씀)
    → {(방식, 층 위치, alpha): DataFrame(시험 장소 × 타깃)}"""
    if configs is None:
        configs = [(p, li) for p in POOLS for li in range(feats[p].shape[1])]
    rows = np.array([p in fold for p in row_place])
    rp = [p for p in row_place if p in fold]
    f_rows = np.array([fold[p] for p in rp])
    trn = trainable[rows] if trainable is not None else np.ones(len(rp), dtype=bool)
    ev = f_rows >= 0
    rp_ev = [p for p, e in zip(rp, ev) if e]
    Yr = Y[rows]
    out = {}
    for pool, li in configs:
        X = feats[pool][rows, li].astype(np.float64)
        pred = {a: np.zeros((len(rp), len(TARGETS))) for a in alphas}
        for f in sorted(set(f_rows[ev])):
            te = f_rows == f
            tr = ~te & trn
            for a, pr in ridge_multi(X[tr], Yr[tr], X[te], alphas).items():
                pred[a][te] = pr
        for a in alphas:
            out[(pool, li, a)] = by_place(pred[a][ev], rp_ev)
    return out


def mae_of(df, true_df, target):
    return (df[target] - true_df.loc[df.index, target]).abs().mean()


def ens_grid(grid, configs, alphas):
    """configs 예측을 평균한 앙상블 → {alpha: DataFrame}"""
    return {a: sum(grid[(p, li, a)] for p, li in configs) / len(configs) for a in alphas}


def choose(grid, true_df, target, method, ens_configs, alphas):
    """교차검증 예측으로 target 의 설정을 고름 → {'configs': [(방식, 층 위치)...], 'alpha': a}"""
    if method == "select":
        pool, li, a = min(grid, key=lambda k: mae_of(grid[k], true_df, target))
        return {"configs": [(pool, li)], "alpha": a}
    ens = ens_grid(grid, ens_configs, alphas)
    a = min(alphas, key=lambda a: mae_of(ens[a], true_df, target))
    return {"configs": list(ens_configs), "alpha": a}


def fit_head(feats, Y, rows, j, spec):
    """설정의 층마다 타깃 j 회귀층 학습 → 회귀층 목록"""
    return [ridge_fit(feats[p][rows, li].astype(np.float64), Y[rows][:, j:j + 1], spec["alpha"])
            for p, li in spec["configs"]]


def predict_head(feats, rows, models, configs):
    """층별 예측의 평균 (층이 하나면 그 층 예측)"""
    return np.mean([ridge_predict(m, feats[p][rows, li].astype(np.float64))[:, 0]
                    for m, (p, li) in zip(models, configs)], axis=0)


def nested_cv(feats, Y, row_place, places, extra, group_of, true_df, k, alphas, method, ens_configs, seed=0,
              trainable=None):
    """바깥 겹: 시험 장소 / 안쪽 겹: 남은 장소로 설정 선택 → 시험 장소는 선택에도 안 쓰임
    extra 장소는 언제나 학습 쪽에만 들어감. seed 로 장소 분할을 바꿈 (seed 0 = 예전 결과와 같은 분할)
    반환: 장소별 예측 [장소, 타깃], 겹별 선택, 문장별 예측 [문장 수, 타깃] (시험 안 한 문장은 NaN)"""
    outer = make_folds(places, [group_of[p] for p in places], k, seed=seed)
    pred = pd.DataFrame(index=places, columns=TARGETS, dtype=float)
    rows_pred = np.full((len(row_place), len(TARGETS)), np.nan)
    chosen = []
    configs = None if method == "select" else ens_configs
    for f in range(k):
        tr = [p for p in places if outer[p] != f]
        te = [p for p in places if outer[p] == f]
        inner = make_folds(tr, [group_of[p] for p in tr], k - 1, seed=seed * 100 + f + 1)
        inner.update({p: -1 for p in extra})
        grid = grid_cv(feats, Y, row_place, inner, alphas, configs, trainable)
        tr_set = set(tr) | set(extra)
        rows_tr = np.array([p in tr_set for p in row_place])
        if trainable is not None:
            rows_tr &= trainable
        rows_te = np.array([outer.get(p, -1) == f for p in row_place])
        te_keys = [p for p in row_place if outer.get(p, -1) == f]
        for j, t in enumerate(TARGETS):
            spec = choose(grid, true_df, t, method, ens_configs, alphas)
            pr = predict_head(feats, rows_te, fit_head(feats, Y, rows_tr, j, spec), spec["configs"])
            rows_pred[rows_te, j] = pr
            pred.loc[te, t] = pd.Series(pr).groupby(te_keys).mean().loc[te].values
            chosen.append({"fold": f, "target": t, "configs": spec["configs"], "alpha": spec["alpha"]})
    return pred.loc[places].values, chosen, rows_pred


# ---------------------------------------------------------------- 데이터
def build_dataset(labels, extra="", sentences=""):
    """평가 장소(labels) · 학습 전용 장소(extra) · 학습 전용 문장(sentences, make_scorer_sentences.py) 를 묶는다.
    반환 dict
      places: 평가 장소 (교차검증 시험 단위) / extra: 늘 학습 쪽인 묶음 이름 (학습 전용 장소 + '#공식N' 문장 묶음)
      true_df: 묶음별 정답 / group_of: 묶음 → 그룹 / items: (문장, 장소 시작, 장소 길이)
      row_place: 문장별 묶음 이름 / Y: 문장별 정답 / trainable: 문장별 학습 사용 여부 (test 전용 형태 패턴은 False)"""
    read = lambda s: pd.concat([pd.read_csv(f, encoding="utf-8-sig") for f in s.split(",") if f]) if s else None
    lab, ext, sen = read(labels), read(extra), read(sentences)
    ext = ext if ext is not None else lab.iloc[:0]
    places, extra_places = lab["place"].tolist(), ext["place"].tolist()
    assert not set(places) & set(extra_places), "평가 장소와 --extra 장소가 겹침"
    keyed = pd.concat([lab, ext])[["place", "group"] + TARGETS]
    items, row_place, Y, trainable = [], [], [], []
    for p, rel, risk in zip(keyed["place"], keyed["relevance"], keyed["risk"]):
        for pat in PATTERNS:
            t, ps = render(pat, p)
            items.append((t, ps, len(p)))
            row_place.append(p)
            Y.append((rel, risk))
            trainable.append(pat not in HELDOUT_PATTERNS)
    pseudo = []
    if sen is not None:
        assert not set(sen["key"]) & set(places), "학습 문장에 평가 장소가 섞임"
        lab_of = keyed.set_index("place")
        for r in sen.itertuples():
            if not str(r.key).startswith("#"):
                assert r.key in lab_of.index and (r.relevance, r.risk) == tuple(lab_of.loc[r.key, TARGETS]), r.key
            else:
                pseudo.append({"place": r.key, "group": r.group, "relevance": r.relevance, "risk": r.risk})
            items.append((r.text, int(r.place_start), int(r.place_len)))
            row_place.append(r.key)
            Y.append((r.relevance, r.risk))
            trainable.append(True)
    n_pat = len(keyed) * len(PATTERNS)
    all_lab = pd.concat([keyed, pd.DataFrame(pseudo, columns=keyed.columns)])
    return {"lab": lab, "places": places, "extra": extra_places + [x["place"] for x in pseudo],
            "kind": ["패턴8"] * n_pat + ["학습문장"] * (len(items) - n_pat),
            "true_df": all_lab.set_index("place")[TARGETS].astype(np.float64),
            "group_of": dict(zip(all_lab["place"], all_lab["group"])), "items": items, "row_place": row_place,
            "Y": np.array(Y, dtype=np.float64), "trainable": np.array(trainable)}


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
    ap.add_argument("--sentences", default="", help="학습에만 쓰는 문장 csv (make_scorer_sentences.py)")
    ap.add_argument("--grid", action="store_true", help="방식·층별 비교표 출력 (36개 설정, 오래 걸림)")
    ap.add_argument("--name", default="scorer_2b_strict")
    ap.add_argument("--method", default="ensemble", choices=["select", "ensemble"],
                    help="ensemble 이 기본 (2026-11-13: select 는 장소별 예측이 분할·라벨 변화에 크게 흔들림)")
    ap.add_argument("--ens_pool", default="last")
    ap.add_argument("--ens_layers", default="9,11,13,15,17,19,21", help="ensemble 에 쓰는 층 번호")
    ap.add_argument("--repeats", type=int, default=1, help="장소 분할을 바꿔 중첩 교차검증 반복 횟수")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--alphas", default="1,10,100,1000,10000", help="릿지 규제 세기 후보")
    ap.add_argument("--exec_rel", type=float, default=0.9, help="실행: 관계율 이 값 이상 + 위험률 risk_max 미만")
    ap.add_argument("--ask_rel", type=float, default=0.5, help="확인 질문: 관계율 이 값 이상. 그 밖은 거부")
    ap.add_argument("--risk_max", type=float, default=0.5)
    ap.add_argument("--high_risk", type=float, default=0.8, help="위험 구역 지표 기준 (정답 위험률 이 값 이상)")
    args = ap.parse_args()
    alphas = [float(a) for a in args.alphas.split(",")]
    np.random.seed(0)

    ds = build_dataset(args.labels, args.extra, args.sentences)
    lab, places, extra, true_df, group_of = ds["lab"], ds["places"], ds["extra"], ds["true_df"], ds["group_of"]
    items, row_place, Y, trainable = ds["items"], ds["row_place"], ds["Y"], ds["trainable"]
    true = true_df.loc[places].values
    print(f"평가 장소 {len(places)}곳 + 학습 전용 묶음 {len(extra)}개 → 문장 {len(items)}개 "
          f"(학습에 쓰는 문장 {int(trainable.sum())}, test 전용 형태 패턴 {int((~trainable).sum())}개는 예측만)")

    feats, layers = load_or_extract(args, items)
    ens_configs = [(args.ens_pool, layers.index(int(l))) for l in args.ens_layers.split(",")]
    cfg_name = lambda cs: "+".join(f"{p}{layers[li]}" for p, li in cs) if len(cs) <= 2 else \
        f"{cs[0][0]}{layers[cs[0][1]]}~{layers[cs[-1][1]]} 평균({len(cs)}층)"
    print(f"방식: {args.method}" + (f" ({cfg_name(ens_configs)})" if args.method == "ensemble" else ""))

    # 기준선: 학습 장소들의 평균값만 예측
    fold = make_folds(places, lab["group"].tolist(), args.folds)
    fold.update({p: -1 for p in extra})
    base = np.zeros_like(true)
    for f in range(args.folds):
        te = np.array([fold[p] == f for p in places])
        base[te] = true_df.loc[[p for p in places + extra if fold[p] != f]].values.mean(0)
    bmae = np.abs(base - true).mean(0)
    print(f"\n기준선(평균값만 예측) MAE: 관계율 {bmae[0]:.3f} / 위험률 {bmae[1]:.3f}")

    # 1) 방식·층별 비교 (참고용, --grid. 같은 교차검증으로 고르므로 최솟값은 약간 낙관적)
    t0 = time.time()
    full = args.grid or args.method == "select"
    grid = grid_cv(feats, Y, row_place, fold, alphas, None if full else ens_configs, trainable)
    table = []
    if full:
        print(f"\n{'방식':<6}{'층':>3} | {'관계율 MAE':>10}{'순위상관':>8} | {'위험률 MAE':>10}{'순위상관':>8}")
    for pool in (POOLS if full else []):
        for li, l in enumerate(layers):
            row = {"pool": pool, "layer": l}
            for t in TARGETS:
                k = min([(pool, li, a) for a in alphas], key=lambda kk: mae_of(grid[kk], true_df, t))
                pr = grid[k].loc[places, t].values
                row[t] = {"alpha": k[2], "mae": float(np.abs(pr - true_df.loc[places, t]).mean()),
                          "rho": spearman(pr, true_df.loc[places, t].values)}
            table.append(row)
            r, k = row["relevance"], row["risk"]
            print(f"{pool:<6}{l:>3} | {r['mae']:>10.3f}{r['rho']:>8.3f} | {k['mae']:>10.3f}{k['rho']:>8.3f}")
    print(f"  교차검증 ({time.time() - t0:.0f}초)")
    selected = {}                               # 최종 회귀층 설정 (전체 평가 장소의 교차검증으로 고름)
    for t in TARGETS:
        spec = choose(grid, true_df, t, args.method, ens_configs, alphas)
        pr = (grid[(*spec["configs"][0], spec["alpha"])] if args.method == "select"
              else ens_grid(grid, spec["configs"], [spec["alpha"]])[spec["alpha"]]).loc[places, t].values
        selected[t] = {**spec, "mae": float(np.abs(pr - true_df.loc[places, t]).mean()),
                       "rho": spearman(pr, true_df.loc[places, t].values)}
        print(f"  {t} 최종 설정: {cfg_name(spec['configs'])} / alpha {spec['alpha']:.0f}"
              f" → MAE {selected[t]['mae']:.3f}, 순위상관 {selected[t]['rho']:.3f}")

    # 2) 중첩 교차검증 (정직한 일반화 수치). 반복하면 장소 분할만 바꿈
    t0 = time.time()
    preds, chosen_all = [], []
    for r in range(args.repeats):
        p, ch, _ = nested_cv(feats, Y, row_place, places, extra, group_of, true_df, args.folds, alphas,
                             args.method, ens_configs, seed=r, trainable=trainable)
        preds.append(p)
        chosen_all.append(ch)
        print(f"  중첩 교차검증 {r + 1}/{args.repeats}", end="\r")
    print()
    preds = np.array(preds)                                 # [반복, 장소, 타깃]
    maes = np.abs(preds - true[None]).mean(1)               # [반복, 타깃]
    rhos = np.array([[spearman(p[:, j], true[:, j]) for j in range(len(TARGETS))] for p in preds])
    pred = preds.mean(0)
    print(f"중첩 교차검증 ({time.time() - t0:.0f}초, {args.repeats}회) MAE: 관계율 {maes[:, 0].mean():.3f} / 위험률 {maes[:, 1].mean():.3f}"
          f"  순위상관: {rhos[:, 0].mean():.3f} / {rhos[:, 1].mean():.3f}  (기준선 MAE {bmae[0]:.3f} / {bmae[1]:.3f})")
    for t in TARGETS:
        cs = [cfg_name(c["configs"]) + (f"·a{c['alpha']:.0f}" if args.method == "ensemble" else "")
              for c in chosen_all[0] if c["target"] == t]
        print(f"  겹별 선택 ({t}, 1회차): {', '.join(cs)}")
    spread = preds.max(0) - preds.min(0)                    # [장소, 타깃] 분할에 따른 예측 폭
    th = (args.exec_rel, args.ask_rel, args.risk_max)
    acts = np.array([[decide(r_, k_, *th) for r_, k_ in p] for p in preds])    # [반복, 장소]
    flips = [places[i] for i in range(len(places)) if len(set(acts[:, i])) > 1]
    if args.repeats > 1:
        print(f"  반복 간 MAE 표준편차: 관계율 {maes[:, 0].std():.3f} / 위험률 {maes[:, 1].std():.3f}")
        for j, t in enumerate(TARGETS):
            i = int(spread[:, j].argmax())
            print(f"  장소별 예측 폭 ({t}): 평균 {spread[:, j].mean():.3f}, 최대 {spread[i, j]:.3f} ({places[i]})")
        print(f"  분할에 따라 행동이 바뀌는 장소 {len(flips)}곳{': ' + ', '.join(flips) if flips else ''}")

    df = lab[["place", "group"] + TARGETS].copy()
    df["pred_relevance"], df["pred_risk"] = pred[:, 0].round(3), pred[:, 1].round(3)
    df["err_relevance"] = (df["pred_relevance"] - df["relevance"]).round(3)
    df["err_risk"] = (df["pred_risk"] - df["risk"]).round(3)
    if args.repeats > 1:
        df["spread_relevance"], df["spread_risk"] = spread[:, 0].round(3), spread[:, 1].round(3)

    # 3) 위험 구역 지표: 정답 위험률이 높은 곳을 얼마나 낮게 보는가 (안전에 중요한 쪽의 오류)
    hi = df[df["risk"] >= args.high_risk]
    hi_low = hi[hi["pred_risk"] < args.risk_max]
    slope = float(np.polyfit(df["risk"], df["pred_risk"], 1)[0])
    under = (df["risk"] - df["pred_risk"]).clip(lower=0)
    print(f"\n위험 구역 (정답 위험률 ≥ {args.high_risk}) {len(hi)}곳: 정답 평균 {hi['risk'].mean():.2f} → 예측 평균 "
          f"{hi['pred_risk'].mean():.2f}, 예측이 {args.risk_max} 미만인 곳 {len(hi_low)}곳"
          f"{' (' + ', '.join(hi_low['place']) + ')' if len(hi_low) else ''}")
    print(f"  위험률 예측 기울기 {slope:.2f}, 과소평가 평균 {under.mean():.3f}, 0.2 이상 과소평가 {int((under >= 0.2).sum())}곳")

    # 4) 행동 판단 시뮬레이션 (라벨 구간으로 정한 목표 행동 vs 예측 점수에 기준값을 적용한 행동)
    df["action"] = [decide(r, k, *LABEL_BANDS) for r, k in zip(df["relevance"], df["risk"])]
    df["pred_action"] = [decide(r, k, *th) for r, k in zip(df["pred_relevance"], df["pred_risk"])]
    conf = pd.crosstab(df["action"], df["pred_action"]).reindex(
        index=["실행", "확인", "거부"], columns=["실행", "확인", "거부"], fill_value=0)
    print(f"\n행동 판단 (예측에 적용한 기준값 — 실행: 관계율 ≥ {th[0]} 이고 위험률 < {th[2]} / 확인: 관계율 ≥ {th[1]} / 그 밖 거부)")
    print(f"  행 = 라벨 구간으로 정한 목표 행동 {LABEL_BANDS}, 열 = 예측 점수로 정한 행동")
    print("  " + conf.to_string().replace("\n", "\n  "))
    acc = float((df["action"] == df["pred_action"]).mean())
    wrong_exec = df[(df["pred_action"] == "실행") & (df["action"] != "실행")]
    wrong_reject = df[(df["action"] == "거부") & (df["pred_action"] != "거부")]
    print(f"  일치율 {acc:.1%}, 잘못 '실행' {len(wrong_exec)}곳"
          f"{' (' + ', '.join(wrong_exec['place']) + ')' if len(wrong_exec) else ''}, "
          f"거부할 곳을 거부 안 함 {len(wrong_reject)}곳")
    if args.repeats > 1:
        accs = [(acts[r] == df["action"].values).mean() for r in range(args.repeats)]
        print(f"  회차별 일치율 {min(accs):.1%} ~ {max(accs):.1%}")

    print("\n그룹별 평균 절대 오차 (관계율 / 위험률):")
    for g, sub in df.groupby("group"):
        print(f"  {g:<20} {sub['err_relevance'].abs().mean():.3f} / {sub['err_risk'].abs().mean():.3f}  ({len(sub)}곳)")
    worst = df.assign(e=df["err_risk"].abs()).nlargest(8, "e")
    print("\n위험률 오차가 큰 장소:")
    for _, r in worst.iterrows():
        print(f"  {r['place']:<12} 위험률 {r['risk']:.2f}→{r['pred_risk']:.2f}  관계율 {r['relevance']:.2f}→{r['pred_relevance']:.2f}")

    os.makedirs("results", exist_ok=True)
    df.to_csv(f"results/{args.name}.csv", index=False, encoding="utf-8-sig")

    # 최종 회귀층: 타깃마다 고른 설정으로 전체 데이터(평가 + 학습 전용)를 다시 학습해서 저장
    heads = {}
    for j, t in enumerate(TARGETS):
        s = selected[t]
        models = fit_head(feats, Y, trainable, j, s)          # test 전용 형태 패턴 문장은 빼고 학습
        heads[t] = {"configs": [(p, layers[li]) for p, li in s["configs"]], "alpha": s["alpha"],
                    "models": [{k: torch.tensor(v) for k, v in m.items()} for m in models]}
    os.makedirs("outputs/scorer", exist_ok=True)
    torch.save({"model": args.model, "prompt": args.prompt, "targets": TARGETS, "method": args.method,
                "heads": heads}, f"outputs/scorer/{args.name}.pt")
    named = lambda cs: [(p, layers[li]) for p, li in cs]
    summary = {
        "model": args.model, "prompt": args.prompt, "labels": args.labels, "extra": args.extra,
        "sentences": args.sentences, "n_rows": len(row_place), "n_trainable": int(trainable.sum()),
        "heldout_patterns": sorted(HELDOUT_PATTERNS),
        "method": args.method, "ens_configs": named(ens_configs) if args.method == "ensemble" else None,
        "n_places": len(places), "n_extra": len(extra), "baseline_mae": bmae.tolist(), "repeats": args.repeats,
        "nested": {"mae": maes.mean(0).tolist(), "mae_std": maes.std(0).tolist(), "rho": rhos.mean(0).tolist(),
                   "spread_mean": spread.mean(0).tolist(), "spread_max": spread.max(0).tolist(),
                   "action_flips": flips,
                   "chosen": [[{**c, "configs": named(c["configs"])} for c in ch] for ch in chosen_all]},
        "selected": {t: {**s, "configs": named(s["configs"])} for t, s in selected.items()},
        "high_risk": {"threshold": args.high_risk, "n": len(hi), "true_mean": float(hi["risk"].mean()),
                      "pred_mean": float(hi["pred_risk"].mean()), "pred_below_risk_max": hi_low["place"].tolist()},
        "risk_slope": slope, "under_mean": float(under.mean()), "under_02": int((under >= 0.2).sum()),
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
