"""test 세트로 모델 정확도 측정 (파인튜닝 전 기준점 / 파인튜닝 후 비교 모두 이 스크립트로).

실행 예:
  python evaluate.py --model Qwen/Qwen3.5-0.8B
  python evaluate.py --model Qwen/Qwen3.5-0.8B --limit 50        # 빠르게 50개만
  python evaluate.py --model outputs/lora_qwen35_08b --name ft    # 파인튜닝한 모델 (나중에)

지표:
  - JSON 유효율: 출력에서 JSON 을 꺼내 파싱할 수 있는 비율
  - 정확도(완전 일치): plan 과 error 가 정답과 완전히 같은 비율
  - 거부 정확도: 거부해야 하는 문장(없는 위치·할 수 없는 일)을 맞는 error 로 거부한 비율
  - 위 지표를 '학습 때 본 패턴' / '처음 보는 패턴(held-out)' 으로 나눠서 출력
결과: results/<이름>.jsonl 에 문장별 예측 저장 (틀린 것 직접 확인용)
"""
import argparse
import json
import os
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from schema import SYSTEM_PROMPT


def load_model(model_id):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    kwargs = dict(dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except (ValueError, KeyError):
        # Qwen3.5 처럼 이미지 입력도 받는 모델은 CausalLM 클래스로 안 열릴 수 있어서 대체 클래스로 시도
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    model.eval()
    tok.padding_side = "left"                  # 배치 생성 시 왼쪽을 채워야 문장 끝이 맞춰짐
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


def build_prompt(tok, text):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}]
    # enable_thinking=False: 사고 모드가 있는 모델(Qwen 계열)은 끄고 바로 답하게 함. 없는 모델은 무시됨
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)


def extract_json(text):
    """출력에서 첫 '{' 부터 마지막 '}' 까지 잘라 파싱. 실패하면 None"""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)    # 사고 모드 흔적 제거
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def normalize(obj):
    """비교용 정규화: plan 리스트와 error 만 본다"""
    if not isinstance(obj, dict):
        return None
    plan = obj.get("plan", [])
    if not isinstance(plan, list):
        return None
    return {"plan": plan, "error": obj.get("error")}


def summarize(rows):
    n = len(rows)
    if n == 0:
        return "0개"
    valid = sum(r["valid"] for r in rows)
    correct = sum(r["correct"] for r in rows)
    rej = [r for r in rows if json.loads(r["output"]).get("error")]
    rej_ok = sum(r["correct"] for r in rej)
    s = f"{n}개 | JSON 유효 {valid / n:.1%} | 정확도 {correct / n:.1%}"
    if rej:
        s += f" | 거부 정확도 {rej_ok / len(rej):.1%} ({len(rej)}개)"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Hugging Face 모델 이름 또는 로컬 경로")
    ap.add_argument("--name", default=None, help="결과 파일 이름 (기본: 모델 이름)")
    ap.add_argument("--test", default="data/test.jsonl")
    ap.add_argument("--limit", type=int, default=None, help="앞에서 N개만 평가")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.test, encoding="utf-8")]
    if args.limit:
        # 본 패턴/처음 보는 패턴이 고르게 섞이도록 앞뒤에서 반씩
        half = args.limit // 2
        rows = rows[:half] + rows[-(args.limit - half):]

    print(f"모델 불러오는 중: {args.model}")
    model, tok = load_model(args.model)
    print(f"파라미터: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    t0 = time.time()
    for i in range(0, len(rows), args.batch):
        batch = rows[i:i + args.batch]
        prompts = [build_prompt(tok, r["input"]) for r in batch]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=256, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        for r, ids in zip(batch, out):
            gen = tok.decode(ids[enc["input_ids"].shape[1]:], skip_special_tokens=True)
            pred = extract_json(gen)
            r["raw"] = gen
            r["valid"] = pred is not None
            r["correct"] = pred is not None and normalize(pred) == normalize(json.loads(r["output"]))
        print(f"  {min(i + args.batch, len(rows))}/{len(rows)}", end="\r")
    elapsed = time.time() - t0

    name = args.name or args.model.replace("/", "_")
    os.makedirs("results", exist_ok=True)
    with open(f"results/{name}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n\n=== {args.model} ({elapsed:.0f}초, 문장당 {elapsed / len(rows):.2f}초) ===")
    print("전체          :", summarize(rows))
    print("본 패턴       :", summarize([r for r in rows if not r.get("heldout")]))
    print("처음 보는 패턴:", summarize([r for r in rows if r.get("heldout")]))
    wrong = [r for r in rows if not r["correct"]][:3]
    if wrong:
        print("\n틀린 예시:")
        for r in wrong:
            print("  입력:", r["input"])
            print("  정답:", r["output"])
            print("  출력:", r["raw"][:200].replace("\n", " "))
    print(f"\n문장별 결과: results/{name}.jsonl")


if __name__ == "__main__":
    main()
