"""할 수 없는 일 추가 평가 세트(data/probe_unsupported.jsonl) 채점: 행동(실행·확인·거부)과 이유(error)를 나눠 본다.
최종 행동은 infer.CommandBrain 과 같은 경로 (2B 해석 + 점수 모델 + 기준값, --judge 면 9B 판단까지)

채점
  - 정답 행동이 거부 (unsupported·unknown·demo 일부): 행동 = 거부인지 / 이유 = 2B error 가 정답 error(unsupported_command·unknown_location) 와 같은지 (거부했어도 이유가 다르면 따로 셈)
  - 정답 행동이 실행 (control·demo 일부): 행동 = 실행인지 / 2B 계획이 정답과 같은지

실행 (해석 모델과 그 모델로 학습한 점수 모델을 함께 지정):
  python probe_check.py --model outputs/qwen35_2b_v3_strict/merged --scorer outputs/scorer/scorer_2b_v3_sent.pt \
      --thresholds results/thresholds_2b_v3.json --name probe_2b_v3
  python probe_check.py --name probe2_2b_v4      # 기본값 = 서버 기본 구성 (2026-11-15 부터 v4)
결과: results/<name>.jsonl (문장별), results/<name>.json (요약)
"""
import argparse
import collections
import json

from evaluate import normalize
from infer import CommandBrain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default="data/probe_unsupported.jsonl")
    ap.add_argument("--model", default="outputs/qwen35_2b_v4_strict/merged")
    ap.add_argument("--prompt", default="strict")
    ap.add_argument("--scorer", default="outputs/scorer/scorer_2b_v4_sent.pt")
    ap.add_argument("--thresholds", default="results/thresholds_2b_v4.json")
    ap.add_argument("--judge", action="store_true", help="'확인' 을 9B 가 판단 (Ollama 필요)")
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.probe, encoding="utf-8")]
    brain = CommandBrain(model=args.model, prompt=args.prompt, scorer=args.scorer, thresholds=args.thresholds,
                         judge=args.judge)
    for i, r in enumerate(rows):
        res = brain(r["input"])
        exp = json.loads(r["output"])
        r.update({k: res[k] for k in ["action", "first_action", "plan", "error", "relevance", "risk", "reason",
                                      "message", "judge"]})
        r["action_ok"] = res["action"] == r["expected_action"]
        if r["expected_action"] == "실행":
            r["plan_ok"] = normalize({"plan": res["plan"], "error": res["error"]}) == normalize(exp)
        else:
            r["reason_ok"] = res["error"] == exp.get("error")
        print(f"  {i + 1}/{len(rows)}", end="\r")
    print()

    summary = {"args": vars(args), "groups": {}}
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["kind"], r["template"], r["heldout"])].append(r)
    print(f"{'종류':12s} {'템플릿':16s} {'패턴':10s} {'수':>3s}  {'행동 맞음':>8s}  {'이유·계획 맞음':>12s}  최종 행동 분포")
    for (kind, tmpl, held), g in sorted(groups.items(), key=lambda x: (x[0][0] != "unsupported", x[0][1], x[0][2])):
        n = len(g)
        a = sum(r["action_ok"] for r in g)
        b = sum(r.get("reason_ok", r.get("plan_ok", False)) for r in g)
        dist = dict(collections.Counter(r["action"] for r in g))
        print(f"{kind:12s} {tmpl:16s} {'처음 보는' if held else '본 패턴':10s} {n:3d}  {a:3d} ({a / n:4.0%})  {b:6d} ({b / n:4.0%})  {dist}")
        summary["groups"][f"{kind}/{tmpl}/{'heldout' if held else 'seen'}"] = {
            "n": n, "action_ok": a, "reason_or_plan_ok": b, "actions": dist}
    for kind in ["unsupported", "unknown", "control", "demo"]:
        g = [r for r in rows if r["kind"] == kind]
        if g:
            a = sum(r["action_ok"] for r in g)
            b = sum(r.get("reason_ok", r.get("plan_ok", False)) for r in g)
            summary[kind] = {"n": len(g), "action_ok": a, "reason_or_plan_ok": b}
            print(f"\n[{kind}] {len(g)}개: 행동 맞음 {a} ({a / len(g):.1%}), "
                  f"{'계획' if kind == 'control' else '이유·계획' if kind == 'demo' else '이유(error)'} 맞음 {b} ({b / len(g):.1%})")
    for r in [r for r in rows if r["kind"] == "demo"]:
        ok = "맞음" if r["action_ok"] and r.get("reason_ok", r.get("plan_ok")) else "틀림"
        print(f"  시연 [{ok}] {r['input']} → {r['action']} (정답 {r['expected_action']}, error {r['error']}, "
              f"계획 {[str(st.get('skill')) + (f"({st.get('target')})" if st.get('target') else '') for st in r['plan'] if isinstance(st, dict)]}, "
              f"관계율 {r['relevance']} · 위험률 {r['risk']}) {r['message']}")

    with open(f"results/{args.name}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(f"results/{args.name}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(f"\n저장: results/{args.name}.jsonl, results/{args.name}.json")


if __name__ == "__main__":
    main()
