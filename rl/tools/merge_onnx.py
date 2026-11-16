"""학습된 ML-Agents 모델(.onnx)을 Unity 에서 쓸 수 있게 한 파일로 합친다.

배경: mlagents 1.1.0 + torch 2.11 은 새 내보내기 방식(dynamo)을 써서 가중치를 별도 파일(.onnx.data)로 저장한다.
     결과 폴더의 GoTo.onnx 는 그래프만 복사된 것이라(가중치 .data 는 안 따라옴) 그대로 Unity 에 넣으면 안 된다.
     → 마지막 스텝의 GoTo-<step>.onnx 를 가중치와 함께 읽어 한 파일로 저장한다.

실행 (conda 환경 mlagents, 저장소 루트에서):
  python rl/tools/merge_onnx.py --run rl/results/goto_v1 --behavior GoTo
  → unity/WarehouseRobot/Assets/02Resources/00Component/Models/GoTo.onnx
"""
import argparse
import glob
import os
import re

import onnx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="rl/results/goto_v1")
    ap.add_argument("--behavior", default="GoTo")
    ap.add_argument("--step", type=int, default=None, help="특정 스텝 체크포인트 (기본: 가장 마지막)")
    ap.add_argument("--out_dir", default="unity/WarehouseRobot/Assets/02Resources/00Component/Models")
    args = ap.parse_args()

    files = glob.glob(os.path.join(args.run, args.behavior, f"{args.behavior}-*.onnx"))
    steps = {int(re.search(r"-(\d+)\.onnx$", f).group(1)): f for f in files}
    if not steps:
        raise SystemExit(f"{args.run}/{args.behavior} 에 {args.behavior}-<step>.onnx 가 없습니다")
    step = args.step if args.step is not None else max(steps)
    src = steps[step]

    model = onnx.load(src)   # 같은 폴더의 .onnx.data 에서 가중치를 함께 읽음
    onnx.checker.check_model(model)
    opset = [o.version for o in model.opset_import if o.domain in ("", "ai.onnx")]
    print(f"원본: {src} (스텝 {step}, opset {opset})")
    print("입력:", [(i.name, [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim]) for i in model.graph.input])
    print("출력:", [o.name for o in model.graph.output])

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"{args.behavior}.onnx")
    onnx.save(model, out, save_as_external_data=False)
    print(f"저장: {out} ({os.path.getsize(out) / 1024:.0f} KB, 가중치 포함 한 파일)")


if __name__ == "__main__":
    main()
