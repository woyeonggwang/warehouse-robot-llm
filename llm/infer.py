"""LLM 파트 최종 진입점: 한국어 명령 한 문장 → 최종 행동(실행·확인·거부)과 기술 순서 JSON.
Unity 연결 단계에서 이 모듈을 직접 쓰거나 --serve 로 띄운 로컬 HTTP 서버를 호출한다.

구성 (2026-11-13 확정, 2026-11-15 해석 모델을 v4 로 교체)
  - 해석 모델: outputs/qwen35_2b_v4_strict/merged (Qwen3.5-2B, 데이터 v4 = v2 + 할 수 없는 일 4유형 + 꾸밈말 이동, strict 프롬프트)
  - 점수 모델: outputs/scorer/scorer_2b_v4_sent.pt (해석 모델 내부 표현 위 릿지 회귀, last 9~21층 평균)
  - 기준값: results/thresholds_2b_v4.json
  - 이전 구성 v3 로 띄우기: --model outputs/qwen35_2b_v3_strict/merged --scorer outputs/scorer/scorer_2b_v3_sent.pt
                            --thresholds results/thresholds_2b_v3.json
  - 최종 행동 규칙: e2e_check.final_action 과 같음
      unsupported_command → 거부 / follow 대상이 me·worker 가 아님 → 거부 / 정지·대기·따라가기만 있는 계획 → 실행 /
      그 밖은 점수 모델: 실행(해석 모델 계획이 있을 때) · 확인 · 거부
  - '확인' 판단 (2026-11-15): 확인이 나오면 사람에게 묻지 않고 9B(judge.py, Ollama qwen3.5:9b, 컨텍스트 4096)가
    실행·거부로 바꿈. 실행할 계획이 없으면 9B 를 부르지 않고 거부. Ollama 에 연결이 안 되면 '확인' 그대로 둠 (사람이 답함)

출력 JSON
  {"input", "action": "실행"|"확인"|"거부" (최종), "first_action": 9B 판단 전 행동, "judge": 9B 판단 {decision, reason, latency_ms} 또는 null,
   "plan": 해석 모델 계획 (확인이면 사용자가 '네' 할 때 실행할 계획),
   "error": 해석 모델 error, "relevance", "risk", "reason", "message": 사용자에게 보여줄 문장, "latency_ms" (9B 포함)}

실행:
  python infer.py --serve --port 8765        # HTTP: POST /command  본문 {"text": "..."} → 위 JSON (127.0.0.1 에서만 받음)
                                             # 시작할 때 9B 를 올리고(컨텍스트 4096, 계속 유지), Ctrl+C 로 끄면 9B 도 내림
  python infer.py "하역장으로 가"            # 한 문장
  python infer.py                            # 대화형 (빈 줄이면 종료)
  --no_judge                                 # 9B 없이 (확인은 확인 그대로)
"""
import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import torch

from e2e_check import bad_follow, final_action, score
from evaluate import build_prompt, extract_json, load_model
from judge import Judge, executable
from make_data import ro
from schema import PROMPTS, SHELVES, SPECIAL_LOCATIONS
from train_scorer import extract

DISPLAY = {loc: names[0] for loc, names in SPECIAL_LOCATIONS.items()} | {s: f"선반 {s}" for s in SHELVES}


class CommandBrain:
    def __init__(self, model="outputs/qwen35_2b_v4_strict/merged", prompt="strict",
                 scorer="outputs/scorer/scorer_2b_v4_sent.pt", thresholds="results/thresholds_2b_v4.json", judge=True):
        self.judge = Judge() if judge else None
        self.scorer = torch.load(scorer, weights_only=False)
        assert self.scorer["model"] == model and self.scorer["prompt"] == prompt, "점수 모델과 해석 모델·프롬프트가 다름"
        t = json.load(open(thresholds, encoding="utf-8"))["thresholds"]
        self.th = (t["exec_rel"], t["ask_rel"], t["risk_max"])
        self.system = PROMPTS[prompt]
        self.model, self.tok = load_model(model)

    @torch.no_grad()
    def __call__(self, text):
        t0 = time.time()
        feats, layers = extract(self.model, self.tok, [(text, 0, len(text))], self.system, verbose=False)
        rel, risk = (float(x) for x in score(self.scorer, feats, layers)[0])
        enc = self.tok([build_prompt(self.tok, text, self.system)], return_tensors="pt",
                       add_special_tokens=False).to("cuda")
        gen = self.model.generate(**enc, max_new_tokens=256, do_sample=False, pad_token_id=self.tok.pad_token_id)
        out = extract_json(self.tok.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True))
        action, reason = final_action(out, rel, risk, self.th)
        plan = out.get("plan", []) if isinstance(out, dict) else []
        err = out.get("error") if isinstance(out, dict) else "parse_error"
        first, judged = action, None
        if action == "확인" and self.judge is not None:
            if not executable(out):
                judged = {"decision": "거부", "reason": "실행할 계획이 없어 9B 를 부르지 않음", "latency_ms": 0}
            else:
                try:
                    judged = self.judge(text, plan, rel, risk)
                except Exception as e:      # Ollama 가 꺼져 있거나 응답 오류 → 확인 그대로 (사람이 답함)
                    judged = {"decision": None, "reason": f"9B 판단 실패: {e}", "latency_ms": 0}
            if judged["decision"]:
                action = judged["decision"]
        return {"input": text, "action": action, "first_action": first, "judge": judged, "plan": plan, "error": err,
                "relevance": round(rel, 3), "risk": round(risk, 3), "reason": reason,
                "message": self.message(action, out, rel, judged), "latency_ms": int((time.time() - t0) * 1000)}

    @staticmethod
    def message(action, out, rel, judged=None):
        """사용자에게 보여줄 문장. 거부는 이유별로 나눔 (2026-11-15): 해석 실패 / 할 수 없는 일 / 목록에 없는 장소 /
        2B 계획은 정상인데 점수 모델이 거부 (관계율 낮음) / 9B 판단"""
        plan = out.get("plan", []) if isinstance(out, dict) and isinstance(out.get("plan"), list) else []
        err = out.get("error") if isinstance(out, dict) else None
        locs = [s.get("target") for s in plan if isinstance(s, dict) and s.get("skill") in ("go_to", "dock")]
        names = list(dict.fromkeys(DISPLAY.get(l, str(l)) for l in locs))
        steps = " → ".join(s.get("skill", "?") for s in plan if isinstance(s, dict))
        if judged and judged.get("decision") == "실행":
            return f"실행합니다 (큰 모델 판단): {steps} — {judged['reason']}"
        if judged and judged.get("decision") == "거부":
            if judged["latency_ms"] == 0:
                return "목록에서 찾을 수 없는 위치라 실행하지 않습니다. 다시 말씀해 주세요."
            return f"실행하지 않습니다 (큰 모델 판단): {judged['reason']}"
        if action == "실행":
            return f"실행합니다: {steps}"
        if action == "확인":
            if names:
                return f"{', '.join(names[:-1] + [ro(names[-1])])} 보면 될까요?"
            return "말씀하신 장소를 찾지 못했습니다. 목록에 있는 위치로 다시 말씀해 주세요."
        if out is None:
            return "명령을 해석하지 못했습니다. 다시 말씀해 주세요."
        if err == "unsupported_command":
            return "할 수 없는 일입니다."
        if bad_follow(out):
            return "따라갈 대상을 알 수 없어 실행하지 않습니다. (따라가기는 나·작업자만 가능)"
        if executable(out):                 # 2B 계획은 정상(목록 안 위치)인데 점수 모델이 거부 = 관계율 < 확인 기준
            where = f"({', '.join(names)}) " if names else ""
            return (f"해석한 계획{where}은 있지만, 명령 속 장소 표현이 창고 업무와 관련이 낮아"
                    f"(관계율 {rel:.2f}) 실행하지 않습니다.")
        return "목록에 없는 장소라 실행하지 않습니다."


def make_server(brain, port):
    """127.0.0.1:port 의 HTTP 서버 (POST /command). 요청을 하나씩 처리 (GPU 모델을 동시에 쓰지 않게)"""
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/command":
                self.send_error(404)
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))).decode("utf-8"))
                res, code = brain(str(body["text"])), 200
            except (ValueError, KeyError) as e:
                res, code = {"error": f"bad_request: {e}"}, 400
            data = json.dumps(res, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, fmt, *args):
            pass

    return HTTPServer(("127.0.0.1", port), Handler)


def serve(brain, port):
    if brain.judge is not None:
        print("9B(qwen3.5:9b) 올리는 중 (컨텍스트 4096, 계속 유지)...")
        brain.judge.load()
    srv = make_server(brain, port)
    print(f"http://127.0.0.1:{port}/command 대기 중 (POST {{\"text\": \"...\"}}, Ctrl+C 로 종료)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if brain.judge is not None:
            brain.judge.unload()
            print("9B 내림")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?", default=None)
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no_judge", action="store_true", help="9B 없이 실행 ('확인' 은 확인 그대로)")
    ap.add_argument("--model", default="outputs/qwen35_2b_v4_strict/merged")
    ap.add_argument("--scorer", default="outputs/scorer/scorer_2b_v4_sent.pt", help="--model 로 학습한 점수 모델")
    ap.add_argument("--thresholds", default="results/thresholds_2b_v4.json")
    args = ap.parse_args()
    brain = CommandBrain(model=args.model, scorer=args.scorer, thresholds=args.thresholds, judge=not args.no_judge)
    if args.serve:
        serve(brain, args.port)
    elif args.text:
        print(json.dumps(brain(args.text), ensure_ascii=False, indent=1))
    else:
        for line in sys.stdin:
            if not line.strip():
                break
            print(json.dumps(brain(line.strip()), ensure_ascii=False))


if __name__ == "__main__":
    main()
