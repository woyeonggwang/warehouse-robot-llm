"""해석 모델 + 점수 모델 연결 점검: 실제 명령 문장으로 두 모델을 함께 돌려 최종 행동(실행·확인·거부)을 확인한다.

최종 행동 규칙 (문장마다)
  1. 해석 모델이 unsupported_command → 거부 (점수 모델 안 씀)
  2. 계획이 정지·대기·따라가기로만 이뤄짐 → 해석 모델 계획대로 실행 (위치와 무관해서 점수 모델 안 씀)
     (처음에는 '계획에 위치가 없으면' 건너뛰었는데, "싣고 있는 거 도크에 내려놔" → [unload] 처럼 해석 모델이 장소를
      빠뜨린 계획이 점수 모델을 거치지 않고 실행되는 구멍이 있어서 좁힘)
  3. 그 밖(위치가 있는 계획, 위치 없는 싣기·내리기, unknown_location 거부)은 점수 모델의 관계율·위험률에 기준값 적용
     - 실행: 해석 모델의 계획이 있고 위치가 모두 목록 안이면 실행. 계획이 없거나 목록 밖 이름이면 확인 (다시 말해 달라고 함)
     - 확인: 해석 모델이 낸 위치를 후보로 물어봄 (없으면 다시 말해 달라고 함)
     - 거부
비교 대상 '해석 모델만': 목록 안 위치로 계획이 있으면 실행, unknown_location 이면 거부 (지금까지의 동작)

데이터
  - 평가 장소 106곳 × 문장 패턴 8개 (848 문장). 점수는 그 장소를 빼고 학습한 모델의 예측 (tune_thresholds.py 가 저장한 문장별 예측)
    목표 행동은 라벨 구간. '실행' 은 계획에 그 장소의 공식 위치(nearest)가 들어 있어야 맞음
  - data/test.jsonl 400 (명령 해석 test): 정상 명령을 점수 모델이 막지 않는지, 거부해야 할 명령이 실행되지 않는지.
    점수는 최종 점수 모델(전체 장소로 학습). 거부 명령의 장소 중 정문·사무실 등은 점수 모델 학습 장소와 겹침

실행: python e2e_check.py
결과: results/<name>.jsonl (문장별), results/<name>.json (요약)
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from evaluate import build_prompt, extract_json, load_model, normalize
from schema import LOCATIONS, PROMPTS
from train_scorer import LABEL_BANDS, TARGETS, decide, extract, ridge_predict

ACTS = ["실행", "확인", "거부"]
NO_PLACE_SKILLS = {"stop", "wait", "follow"}       # 위치와 무관한 기술 (follow 의 target 은 me/worker)
CANDIDATES = [(0.85, 0.5, 0.5), (0.9, 0.5, 0.5), (0.9, 0.5, 0.45), (0.9, 0.5, 0.4), (0.9, 0.5, 0.35), (0.92, 0.5, 0.5)]


def plan_locations(out):
    """계획 속 위치 목록, 계획이 없거나 형식이 틀리면 None"""
    if not isinstance(out, dict) or not isinstance(out.get("plan"), list):
        return None
    return [s.get("target") for s in out["plan"] if isinstance(s, dict) and s.get("skill") in ("go_to", "dock")]


def final_action(out, rel, risk, th):
    """(최종 행동, 이유)"""
    if out is None:
        return "거부", "JSON 파싱 실패"
    err = out.get("error")
    locs = plan_locations(out)
    if err == "unsupported_command":
        return "거부", "할 수 없는 일"
    if not err and locs is not None and out["plan"] and \
            all(isinstance(s, dict) and s.get("skill") in NO_PLACE_SKILLS for s in out["plan"]):
        return "실행", "위치 없는 명령"
    valid_plan = not err and locs is not None and all(l in LOCATIONS for l in locs)
    a = decide(rel, risk, *th)
    if a == "실행":
        return ("실행", "점수 실행") if valid_plan else ("확인", "점수는 실행이지만 해석 모델 계획 없음")
    return a, "점수 " + a


def interp_only(out):
    if out is None:
        return "거부"
    locs = plan_locations(out)
    if out.get("error") or locs is None:
        return "거부"
    return "실행" if all(l in LOCATIONS for l in locs) else "확인"


@torch.no_grad()
def generate(model, tok, texts, system, batch):
    outs = []
    for i in range(0, len(texts), batch):
        prompts = [build_prompt(tok, t, system) for t in texts[i:i + batch]]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        gen = model.generate(**enc, max_new_tokens=256, do_sample=False, pad_token_id=tok.pad_token_id)
        outs += [tok.decode(ids[enc["input_ids"].shape[1]:], skip_special_tokens=True) for ids in gen]
        print(f"  생성 {min(i + batch, len(texts))}/{len(texts)}", end="\r")
    print()
    return outs


def score(scorer, feats, layers):
    """최종 점수 모델로 [문장 수, 2] 예측"""
    out = []
    for t in TARGETS:
        h = scorer["heads"][t]
        ps = [ridge_predict({k: v.numpy() for k, v in m.items()}, feats[p][:, layers.index(l)].astype(np.float64))[:, 0]
              for m, (p, l) in zip(h["models"], h["configs"])]
        out.append(np.mean(ps, 0))
    return np.stack(out, 1)


def show_conf(tgt, pred):
    m = pd.crosstab(pd.Series(tgt, name="목표"), pd.Series(pred, name="최종")).reindex(index=ACTS, columns=ACTS, fill_value=0)
    return "  " + m.to_string().replace("\n", "\n  "), m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="outputs/qwen35_2b_v2_strict/merged")
    ap.add_argument("--prompt", default="strict")
    ap.add_argument("--scorer", default="outputs/scorer/scorer_2b_strict_sent.pt")
    ap.add_argument("--rows", default="results/scorer_2b_strict_sent_rows.csv")
    ap.add_argument("--th", default="file",
                    help="기준값 'file' (results/thresholds.json, 2026-11-13 확정값) 또는 '실행관계율,확인관계율,위험률상한'")
    ap.add_argument("--labels", default="data/labels.csv")
    ap.add_argument("--test", default="data/test.jsonl")
    ap.add_argument("--name", default="e2e_2b_strict")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    if args.th == "file":
        t_ = json.load(open("results/thresholds.json", encoding="utf-8"))["thresholds"]
        th = (t_["exec_rel"], t_["ask_rel"], t_["risk_max"])
    else:
        th = tuple(float(x) for x in args.th.split(","))
    t_ = dict(zip(["exec_rel", "ask_rel", "risk_max"], th))
    scorer = torch.load(args.scorer, weights_only=False)
    assert scorer["model"] == args.model and scorer["prompt"] == args.prompt, "점수 모델과 해석 모델·프롬프트가 다름"
    lab = pd.read_csv(args.labels, encoding="utf-8-sig").set_index("place")
    rows = pd.read_csv(args.rows, encoding="utf-8-sig")
    ev = rows[rows.split == "eval"].reset_index(drop=True)
    test = [json.loads(l) for l in open(args.test, encoding="utf-8")]
    system = PROMPTS[args.prompt]
    print(f"기준값: 실행 관계율 ≥ {th[0]} 이고 위험률 < {th[2]} / 확인 관계율 ≥ {th[1]} / 그 밖 거부")

    print(f"모델 불러오는 중: {args.model}")
    model, tok = load_model(args.model)
    t0 = time.time()
    raws = generate(model, tok, ev.text.tolist() + [r["input"] for r in test], system, args.batch)
    gen_sec = time.time() - t0
    tf, layers = extract(model, tok, [(r["input"], 0, len(r["input"])) for r in test], system)
    del model
    torch.cuda.empty_cache()
    test_scores = score(scorer, tf, layers)
    print(f"생성 {gen_sec:.0f}초 ({len(raws)}문장, 문장당 {gen_sec / len(raws):.2f}초)")

    records, outs = [], []
    # ── 평가 장소 문장
    for i, r in ev.iterrows():
        out = extract_json(raws[i])
        outs.append(out)
        fa, why = final_action(out, r.pred_relevance, r.pred_risk, th)
        locs = plan_locations(out) or []
        tgt = ACTS[int(r.target)]
        near = lab.loc[r.place, "nearest"]
        records.append({"set": "eval", "input": r.text, "place": r.place, "group": r.group, "target": tgt,
                        "relevance": r.relevance, "risk": r.risk, "pred_relevance": r.pred_relevance,
                        "pred_risk": r.pred_risk, "raw": raws[i], "locations": locs, "nearest": near,
                        "final": fa, "why": why, "interp_only": interp_only(out),
                        "loc_ok": near in locs})
    # ── test.jsonl
    for j, r in enumerate(test):
        out = extract_json(raws[len(ev) + j])
        outs.append(out)
        exp = json.loads(r["output"])
        rel, risk = test_scores[j]
        fa, why = final_action(out, rel, risk, th)
        records.append({"set": "test", "input": r["input"], "template": r["template"], "heldout": r["heldout"],
                        "expected_error": exp.get("error"), "pred_relevance": round(float(rel), 4),
                        "pred_risk": round(float(risk), 4), "raw": raws[len(ev) + j], "final": fa, "why": why,
                        "interp_only": interp_only(out),
                        "interp_correct": out is not None and normalize(out) == normalize(exp)})
    os.makedirs("results", exist_ok=True)
    with open(f"results/{args.name}.jsonl", "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = {"thresholds": t_, "scorer": args.scorer}
    E = pd.DataFrame([x for x in records if x["set"] == "eval"])
    print(f"\n=== 평가 장소 106곳 × 8문장 = {len(E)}문장 (점수는 그 장소를 뺀 모델) ===")
    for col, label in [("interp_only", "해석 모델만"), ("final", "해석 + 점수 모델")]:
        s, m = show_conf(E.target, E[col])
        ok_exec = int(((E[col] == "실행") & (E.target == "실행") & E.loc_ok).sum())
        wrong = E[(E[col] == "실행") & (E.target != "실행")]
        danger = wrong[wrong.risk >= 0.5]
        acc = float((E[col] == E.target).mean())
        print(f"\n[{label}] 문장 일치 {acc:.1%}, 공식 위치 바르게 실행 {ok_exec}/{int((E.target == '실행').sum())}, "
              f"잘못 실행 {len(wrong)}문장 ({wrong.place.nunique()}곳, 위험률 0.5 이상 {len(danger)}문장)")
        print(s)
        summary[f"eval_{col}"] = {"agreement": acc, "official_exec_ok": ok_exec, "wrong_exec": len(wrong),
                                  "wrong_exec_places": wrong.place.value_counts().to_dict(),
                                  "danger_exec": len(danger), "confusion": m.to_dict()}
        if col == "final":
            for p, n in wrong.place.value_counts().items():
                w = wrong[wrong.place == p].iloc[0]
                print(f"    잘못 실행: {p} {n}/8 (정답 {w.relevance}/{w.risk}, 예: '{w.input}' → {w.locations})")
            bad_loc = E[(E.final == "실행") & (E.target == "실행") & ~E.loc_ok]
            for _, w in bad_loc.iterrows():
                print(f"    공식 위치인데 다른 곳으로 실행: '{w.input}' → {w.locations} (정답 {w.nearest})")
            miss = E[(E.target == "실행") & (E.final != "실행")]
            print(f"    공식 위치를 실행 안 함 {len(miss)}문장: {miss.place.value_counts().to_dict()}")

    T = pd.DataFrame([x for x in records if x["set"] == "test"])
    normal, unk, uns = T[T.expected_error.isna()], T[T.expected_error == "unknown_location"], \
        T[T.expected_error == "unsupported_command"]
    print(f"\n=== test.jsonl {len(T)}문장 (정상 {len(normal)} / 없는 위치 {len(unk)} / 할 수 없는 일 {len(uns)}) ===")
    blocked = normal[normal.final != "실행"]
    for col, label in [("interp_only", "해석 모델만"), ("final", "해석 + 점수 모델")]:
        n_ok = int(((normal[col] == "실행") & normal.interp_correct).sum())
        unk_exec = int((unk[col] == "실행").sum())
        uns_rej = int((uns[col] == "거부").sum())
        print(f"[{label}] 정상 명령 바르게 실행 {n_ok}/{len(normal)} ({n_ok / len(normal):.1%}), "
              f"없는 위치 명령 실행 {unk_exec}/{len(unk)}, 할 수 없는 일 거부 {uns_rej}/{len(uns)}")
        summary[f"test_{col}"] = {"normal_ok": n_ok, "normal": len(normal), "unknown_exec": unk_exec,
                                  "unknown": len(unk), "unsupported_rejected": uns_rej, "unsupported": len(uns)}
    print(f"  점수 모델이 막은 정상 명령 {len(blocked)}문장 (확인 {int((blocked.final == '확인').sum())}, "
          f"거부 {int((blocked.final == '거부').sum())})")
    for _, w in blocked.head(10).iterrows():
        print(f"    '{w.input}' → {w.final} (관계율 {w.pred_relevance:.2f}, 위험률 {w.pred_risk:.2f})")
    summary["test_blocked"] = blocked[["input", "final", "pred_relevance", "pred_risk"]].to_dict("records")
    normal = normal.assign(multi=normal.template.str.contains(r"\+"))
    for m, label in [(False, "한 작업"), (True, "두 작업 연결")]:
        s = normal[normal.multi == m]
        print(f"  {label} {len(s)}문장: 바르게 실행 {int(((s.final == '실행') & s.interp_correct).sum())} "
              f"(해석 모델만 {int(((s.interp_only == '실행') & s.interp_correct).sum())})")

    # 기준값 후보별 비교: 같은 출력·점수에 기준값만 바꿔 적용
    print("\n=== 기준값 후보별 비교 (같은 출력·점수) ===")
    print(f"  {'기준값':<18} | 평가: 공식 위치 바르게 실행 / 잘못 실행 문장(곳, 위험률≥0.5) | test: 정상 바르게 실행 (한 작업 / 두 작업) / 없는 위치 실행")
    cand = []
    for c in CANDIDATES + ([th] if th not in CANDIDATES else []):
        fe = [final_action(outs[i], x["pred_relevance"], x["pred_risk"], c)[0] for i, x in enumerate(records) if x["set"] == "eval"]
        ft = [final_action(outs[i], x["pred_relevance"], x["pred_risk"], c)[0] for i, x in enumerate(records) if x["set"] == "test"]
        e2, t2 = E.assign(f=fe), T.assign(f=ft)
        ok = int(((e2.f == "실행") & (e2.target == "실행") & e2.loc_ok).sum())
        w = e2[(e2.f == "실행") & (e2.target != "실행")]
        n2 = t2[t2.expected_error.isna()].assign(multi=lambda d: d.template.str.contains(r"\+"))
        good = (n2.f == "실행") & n2.interp_correct
        row = {"th": c, "eval_official_ok": ok, "eval_wrong_exec": len(w), "eval_wrong_places": int(w.place.nunique()),
               "eval_danger": int((w.risk >= 0.5).sum()), "test_normal_ok": int(good.sum()),
               "test_single_ok": int(good[~n2.multi].sum()), "test_multi_ok": int(good[n2.multi].sum()),
               "test_unknown_exec": int((t2[t2.expected_error == "unknown_location"].f == "실행").sum())}
        cand.append(row)
        print(f"  {str(c):<18} | {ok}/{int((E.target == '실행').sum())} / {len(w)}({row['eval_wrong_places']}곳, {row['eval_danger']})"
              f" | {row['test_normal_ok']}/{len(normal)} ({row['test_single_ok']}/{int((~normal.multi).sum())} / "
              f"{row['test_multi_ok']}/{int(normal.multi.sum())}) / {row['test_unknown_exec']}")
    summary["candidates"] = cand
    with open(f"results/{args.name}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(f"\n저장: results/{args.name}.jsonl, results/{args.name}.json")


if __name__ == "__main__":
    main()
