"""거부 집중 점검용 세트: 학습 데이터에 없는 '실제로 있을 법한 장소 이름' 으로만 만든 없는 위치 명령.

실행: python make_probe.py  →  data/probe_reject.jsonl
- 정답은 전부 {"plan": [], "error": "unknown_location"}
- 장소 이름은 make_data.py 의 거부 데이터(2층 사무실·주차장·옥상·휴게실·식당·정문)와 겹치지 않게 골랐다.
  일부(검수대·반품 구역·출입구 등)는 창고에 있을 법해서 모델이 기존 위치(inbound/outbound 등)로 끼워 맞추기 쉬운 이름
- 평가: python evaluate.py --model <모델> --test data/probe_reject.jsonl [--prompt strict]
"""
import json
import random

from make_data import ro, eul

random.seed(7)

PLACES = ["하역장", "경비실", "냉동 창고", "지하 창고", "로비", "회의실", "탈의실", "3층 선반",
          "컨테이너 야드", "반품 구역", "검수대", "출입구", "후문", "분류 라인", "적재장", "자재실"]

PATTERNS = [
    lambda p: f"{ro(p)} 가",
    lambda p: f"{ro(p)} 이동해",
    lambda p: f"{p}에 가 있어",
    lambda p: f"택배 {p}에 내려놔",
    lambda p: f"{p}에서 상자 꺼내서 출고장에 갖다 놔",
    lambda p: f"A1에서 박스 꺼내서 {p}에 갖다 놔",
    lambda p: f"싣고 있는 거 {p}에 내려놔",
    lambda p: f"{p}까지 가 줄래?",
]


def main():
    out = json.dumps({"plan": [], "error": "unknown_location"}, ensure_ascii=False)
    rows = []
    for p in PLACES:
        for pat in random.sample(PATTERNS, 4):          # 장소마다 문장 패턴 4개
            rows.append({"input": pat(p), "output": out, "template": "probe_reject", "heldout": True})
    with open("data/probe_reject.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"data/probe_reject.jsonl: {len(rows)}개")
    for r in rows[:6]:
        print(" ", r["input"])


if __name__ == "__main__":
    main()
