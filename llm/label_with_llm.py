"""큰 로컬 LLM(메인 컴퓨터 역할)으로 장소별 '창고 업무 관계율'·'위험률' 초안 점수를 매긴다.

준비: Ollama 실행 중 + `ollama pull qwen3.5:9b`
실행:
  python label_with_llm.py --limit 5        # 5개만 시험
  python label_with_llm.py                  # data/places.csv 전체
결과: data/labels_draft.csv (엑셀에서 열어 검토·수정 → data/labels.csv 로 저장해서 확정)
- 이미 점수가 있는 장소는 건너뛰므로 중간에 끊겨도 다시 실행하면 이어서 진행
"""
import argparse
import csv
import json
import os
import time
import urllib.request

from schema import LOCATIONS

OLLAMA_URL = "http://localhost:11434/api/chat"

RUBRIC = f"""너는 실내 창고 운반 로봇의 안전 평가자다. 로봇이 아는 공식 위치는 다음뿐이다:
선반 A1~D4, inbound(입고장), outbound(출고장), packing(포장대), charger(충전소).
사용자가 말한 '장소' 하나를 받아서 두 점수를 0.0~1.0 사이 소수 둘째 자리까지 매겨라.

[relevance: 창고 업무 관계율] 그 장소가 창고 물류 작업(입고·보관·피킹·포장·출고·충전)과 얼마나 관련 있는가
- 1.0: 공식 위치와 같은 곳의 다른 이름
- 0.7~0.9: 창고 물류 작업이 실제로 일어나지만 공식 목록에는 없는 곳
- 0.4~0.6: 창고 건물 안이지만 물류 작업과는 간접적인 곳
- 0.1~0.3: 사무·생활 공간
- 0.0: 창고와 무관한 곳

[risk: 위험률] 로봇이 이 명령을 잘못 해석해서 그 장소로(또는 그 장소 대신 다른 곳으로) 짐을 싣고 움직였을 때의 피해 크기
- 0.9~1.0: 사람 부상·추락·화재·감전·설비 파손 가능성이 큼 (계단, 전기실, 사람 밀집 지역 등)
- 0.6~0.8: 고가 물품·온도 관리 물품 손상, 다른 장비 작업 방해
- 0.3~0.5: 물건이 엉뚱한 곳에 놓여 재작업이 필요한 정도
- 0.0~0.2: 거의 피해 없음

[nearest] 공식 위치 중 의미가 가장 가까운 것 하나 (A1~D4, inbound, outbound, packing, charger), 없으면 "none"
[reason] 판단 이유를 한국어 한 문장으로

JSON 으로만 답하라: {{"relevance": 0.0, "risk": 0.0, "nearest": "none", "reason": ""}}"""

SCHEMA = {
    "type": "object",
    "properties": {
        "relevance": {"type": "number"},
        "risk": {"type": "number"},
        "nearest": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["relevance", "risk", "nearest", "reason"],
}


def ask(model, place):
    body = {
        "model": model,
        "messages": [{"role": "system", "content": RUBRIC}, {"role": "user", "content": f"장소: {place}"}],
        "format": SCHEMA,              # Ollama 구조화 출력: 이 JSON 형식으로만 답하게 강제
        "think": False,                # 사고 모드 끔 (점수만 빠르게)
        "stream": False,
        "options": {"temperature": 0},  # 같은 입력이면 같은 점수 (재현성)
    }
    req = urllib.request.Request(OLLAMA_URL, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        content = json.loads(r.read().decode("utf-8"))["message"]["content"]
    out = json.loads(content)
    clip = lambda v: round(min(1.0, max(0.0, float(v))), 2)
    nearest = out.get("nearest", "none")
    if nearest not in LOCATIONS:
        nearest = "none"               # 목록에 없는 이름을 지어내면 none 으로 정리
    return clip(out["relevance"]), clip(out["risk"]), nearest, out.get("reason", "").replace("\n", " ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5:9b")
    ap.add_argument("--places", default="data/places.csv")
    ap.add_argument("--out", default="data/labels_draft.csv")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    places = list(csv.DictReader(open(args.places, encoding="utf-8-sig")))
    if args.limit:
        places = places[:args.limit]

    done = {}
    if os.path.exists(args.out):
        for row in csv.DictReader(open(args.out, encoding="utf-8-sig")):
            done[row["place"]] = row

    fields = ["place", "group", "relevance", "risk", "nearest", "reason", "labeler"]
    t0 = time.time()
    rows = []
    for i, p in enumerate(places, 1):
        if p["place"] in done:
            rows.append(done[p["place"]])
            continue
        rel, risk, near, reason = ask(args.model, p["place"])
        rows.append({"place": p["place"], "group": p["group"], "relevance": rel, "risk": risk,
                     "nearest": near, "reason": reason, "labeler": args.model})
        print(f"[{i}/{len(places)}] {p['place']}: 관계율 {rel}, 위험률 {risk}, 가까운 위치 {near} | {reason}")
        # 한 줄씩 저장해서 중간에 끊겨도 진행분이 남게 함
        with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows + [r for k, r in done.items() if k not in {x["place"] for x in rows}])
    print(f"완료: {args.out} ({time.time() - t0:.0f}초)")


if __name__ == "__main__":
    main()
