"""'확인' 판단기(judge.py, 9B) 점검: 연결 점검(e2e_check.py)에서 '확인' 이 나온 문장을 9B 에 판단시켜 본다.
2B 해석 결과·점수는 results/e2e_2b_strict.jsonl 에 저장된 것을 그대로 씀 (평가 문장 점수는 중첩 교차검증 값) → 2B 를 다시 돌리지 않음

  - 평가 문장 (처음 보는 장소 106곳): 라벨 행동(실행·확인·거부)별로 9B 가 실행한 수, 그중 위험률 0.5 이상
  - test: 정상 명령은 바르게 실행했는지, 없는 위치·할 수 없는 일은 거부했는지

실행: python judge_check.py      결과: results/judge_9b.jsonl, results/judge_9b.json
"""
import argparse
import collections
import json
import time

from judge import Judge, executable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="results/e2e_2b_strict.jsonl")
    ap.add_argument("--out", default="results/judge_9b")
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.rows, encoding="utf-8")]
    rows = [r for r in rows if r["final"] == "확인"]
    judge = Judge()
    judge.load()
    t0 = time.time()
    for i, r in enumerate(rows):
        try:
            out = json.loads(r["raw"])
        except (json.JSONDecodeError, TypeError):
            out = None
        if executable(out):
            j = judge(r["input"], out["plan"], r["pred_relevance"], r["pred_risk"])
        else:
            j = {"decision": "거부", "reason": "실행할 계획 없음 (9B 안 부름)", "latency_ms": 0}
        r["judge"], r["judge_reason"], r["judge_ms"] = j["decision"], j["reason"], j["latency_ms"]
        print(f"  {i + 1}/{len(rows)}", end="\r")
    print(f"\n{len(rows)}문장 {time.time() - t0:.0f}초")
    called = [r for r in rows if r["judge_ms"] > 0]
    ms = sorted(r["judge_ms"] for r in called)
    summary = {"n": len(rows), "judge_called": len(called),
               "latency_ms_median": ms[len(ms) // 2] if ms else None, "sets": {}}
    print(f"9B 호출 {len(called)}문장, 응답 시간 중앙값 {summary['latency_ms_median']} ms")

    ev = [r for r in rows if r["set"] == "eval"]
    print("\n[평가 문장: 처음 보는 장소] 라벨 행동별 9B 판단")
    for tgt in ["실행", "확인", "거부"]:
        g = [r for r in ev if r["target"] == tgt]
        ex = [r for r in g if r["judge"] == "실행"]
        risky = [r for r in ex if r["risk"] >= 0.5]
        summary["sets"][f"eval_target_{tgt}"] = {"n": len(g), "exec": len(ex), "exec_risk_ge_0.5": len(risky)}
        print(f"  라벨 {tgt}: {len(g)}문장 중 9B 실행 {len(ex)} (위험률 0.5 이상 {len(risky)})")
        if tgt != "실행" and ex:
            for p, n in collections.Counter(r["place"] for r in ex).most_common():
                one = next(r for r in ex if r["place"] == p)
                print(f"    실행: {p} {n}문장 (라벨 관계율 {one['relevance']}, 위험률 {one['risk']})")

    te = [r for r in rows if r["set"] == "test"]
    normal = [r for r in te if r["expected_error"] is None]
    bad = [r for r in te if r["expected_error"] is not None]
    n_ok = sum(r["judge"] == "실행" and r["interp_correct"] for r in normal)
    n_wrong_plan = sum(r["judge"] == "실행" and not r["interp_correct"] for r in normal)
    n_bad_exec = sum(r["judge"] == "실행" for r in bad)
    summary["sets"]["test"] = {"normal": len(normal), "normal_exec_correct": n_ok, "normal_exec_wrong_plan": n_wrong_plan,
                               "should_reject": len(bad), "should_reject_exec": n_bad_exec}
    print(f"\n[test] 정상 명령 {len(normal)}문장 중 9B 실행 (계획 맞음) {n_ok}, 실행했지만 계획 틀림 {n_wrong_plan}")
    print(f"       거부해야 할 명령 {len(bad)}문장 중 9B 실행 {n_bad_exec}")
    for r in [r for r in normal if r["judge"] == "거부"][:8]:
        print(f"    거부한 정상 명령: {r['input']}  — {r['judge_reason']}")

    with open(args.out + ".jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(args.out + ".json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(f"\n저장: {args.out}.jsonl, {args.out}.json")


if __name__ == "__main__":
    main()
