"""공식 위치의 다른 이름·경계 장소(data/places_alias.csv)의 검토안을 만든다.

places_alias.csv 는 장소를 만들 때 어느 공식 위치를 가리키는지 정해 두었으므로, 점수(관계율·위험률·nearest)와
근거(note)를 그 파일에 직접 적었다. 9B 초안(label_with_llm.py)은 비교용으로만 남긴다 (draft 열).
  - 공식 위치의 다른 이름: 관계율 1.0 (같은 곳의 다른 이름, 채점 기준표의 1.0 정의. make_data.py 의 선반 표기 6가지 포함) 또는
    0.9 (학습 데이터에 없고 칸 번호를 해석해야 하는 표기 — '3번째 줄 B 선반' 등), 위험률은 확정된 공식 위치와 같은 0.3
  - 경계 장소: 1차 검토 기준 (공식 위치와 기능이 거의 같음 0.8~0.85, 확정된 비슷한 장소와 같은 점수)
  - places_alias.csv 에서 비워 둔 값은 9B 초안 값을 그대로 씀
labeler 열은 'qwen3.5:9b+검토' (2026-11-13 사용자가 초안과 차이가 큰 곳을 검토한 뒤 확정).
학습 전용 데이터: data/labels.csv(평가용 106곳)와 합치지 않고 train_scorer.py --extra 로만 쓴다.

실행: python fix_labels_alias.py data/places_alias.csv data/labels_alias_draft.csv data/labels_alias.csv
"""
import sys

import pandas as pd

from schema import LOCATIONS

spec_path, src, out = sys.argv[1], sys.argv[2], sys.argv[3]
spec = pd.read_csv(spec_path, encoding="utf-8-sig").set_index("place")
d = pd.read_csv(src, encoding="utf-8-sig")
assert set(d.place) == set(spec.index), "초안과 장소 목록이 다름"

d["draft"] = d["relevance"].astype(str) + "/" + d["risk"].astype(str) + "/" + d["nearest"]
draft_rel = d["relevance"].copy()
for col in ["relevance", "risk", "nearest"]:
    v = d.place.map(spec[col])
    d[col] = v.where(v.notna(), d[col])
d["reason"] = d.place.map(spec["note"]).fillna(d["reason"])
d["labeler"] = "qwen3.5:9b+검토"

off = d.group == "official"
assert (d.loc[off, "relevance"] >= 0.9).all() and d.loc[off, "nearest"].isin(LOCATIONS).all()
assert (d.loc[~off, "relevance"] < 0.9).all()
d.to_csv(out, index=False, encoding="utf-8-sig")

print(d.groupby("group")[["relevance", "risk"]].agg(["mean", "min", "max", "count"]).round(2))
gap = d[(d.relevance - draft_rel).abs() >= 0.3]
print(f"\n9B 초안과 관계율이 0.3 이상 다른 곳 {len(gap)}곳 (검토할 때 참고):")
for _, r in gap.iterrows():
    print(f"  {r['place']:<14} 초안 {r['draft']:<22} → {r['relevance']}/{r['risk']}/{r['nearest']}")
print(f"→ {out}")
