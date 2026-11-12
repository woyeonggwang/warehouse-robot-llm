# warehouse-robot-llm — CLAUDE.md

한국어 자연어 명령을 로봇(또는 시뮬레이션 에이전트) 명령 JSON 으로 바꾸는 소형 LLM 을 LoRA 로 파인튜닝하는 프로젝트. 폰·스마트안경 같은 기기에서도 돌아갈 크기(0.5B~1.5B)를 목표로 한다. 구조 (2026-11-12 확정, B안): 소형 LLM 이 한국어 명령을 '기술(skill) 순서' JSON 으로 바꾸고, Unity ML-Agents 로 강화학습한 에이전트가 각 기술을 실행하는 계층 구조. 처음에는 바퀴형 로봇/차량으로 시작 (휴머노이드 보행은 학습 난이도가 높아 보류)
  - 이유: Unity·ML-Agents 경험 활용, 강화학습 + LLM 을 함께 보여주는 포트폴리오, 이 세션의 목표인 소형 LLM 파인튜닝을 그대로 포함
  - 진행 순서: ① LLM 파트(환경, 기술 목록·JSON 설계, 데이터 생성, 파인튜닝) → ② Unity 강화학습 파트 → ③ 연결
  - VLA 연동은 이후 단계 후보
- 시나리오 (2026-11-12 확정): 실내 창고에서 일하는 바퀴형 운반 로봇
  - 이유: 야외(화성 탐사·재난 구조 등)는 지형·날씨 등 변수가 너무 많아짐. 실내 창고는 환경을 통제하기 쉬움
  - 기술 목록·JSON 스키마 확정 (2026-11-12). 기준 파일은 `schema.py` (데이터 생성·학습·평가가 모두 이 파일을 참조)

예시 (형식 확정 전):
```
입력: "빨간 상자 집어서 왼쪽 트레이에 올려놔"
출력: {"task": "pick_and_place", "object": "red_box", "target": "left_tray"}
```

## 저장소 구조 (2026-11-12 결정)
- 저장소 이름: `warehouse-robot-llm` (2026-11-12 결정, 이전 폴더명 robot_cmd_llm)
- LLM 파트와 Unity 강화학습 파트를 한 저장소(모노레포)로 관리
  - 이유: LLM 이 내는 명령 JSON 을 Unity 에이전트가 그대로 실행하므로 명령 체계를 한 커밋으로 같이 관리할 수 있고, 포트폴리오에서 전체 흐름을 저장소 하나로 보여줄 수 있음. 용량 큰 폴더는 .gitignore 로 제외
```
warehouse-robot-llm/
├─ llm/      schema.py, make_data.py, train.py, evaluate.py, data/, results/ (outputs/ 는 올리지 않음)
├─ unity/    Unity 프로젝트
└─ rl/configs/  ML-Agents 학습 설정 (rl/results/ 는 올리지 않음)
```
- .gitignore: Python 캐시, llm/outputs/, unsloth_compiled_cache/, 모델 가중치(*.safetensors 등), rl/results/, Unity 의 Library·Temp·Logs·UserSettings 등
- data/ 는 올림 (작고, 특히 고정 test 세트가 비교 기준이라 보존 필요)

## 작업 규칙 (Claude)
- 시행착오(증상·원인·해결), 확정된 설계 결정과 이유, 버전·환경 고정 사항, 미해결 과제는 이 파일에 기록한다. 확인된 것만 적고, 확인되지 않은 것은 "(확인 필요)"로 표시한다.
- 시행착오에는 개념·환경·설계에서 배울 점이 있는 것만 적는다. 오타 같은 단순 작성 실수는 기록하지 않는다.
- 모델·데이터 선택 이유는 한국어 성능, 라이선스, 학습 도구 지원, VRAM·기기 크기 같은 기술적 기준으로 적는다.
- 백업 파일은 사용자가 요청할 때만 만든다.
- 코드 작성 방식 (2026-11-12 결정): Claude 가 코드 파일을 직접 작성·수정하고, 사용자는 실행·확인·방향 결정을 맡는다.
  - 이유: 라이브러리 조립 위주의 작업이라 손코딩은 비효율적

## 명령 체계 (schema.py)
- 위치: 선반 A1~D4(16칸), inbound(입고장), outbound(출고장), packing(포장대), charger(충전소)
- 물건: box(상자·박스), parcel(택배·소포), tote(바구니·토트)
- 기술: go_to(target), dock(target), follow(target: me|worker) — 강화학습 담당 / load(item), unload, charge, wait(sec), stop — 스크립트로 즉시 처리
  - 바퀴 로봇이 직접 물건을 집는 대신, 선반·충전기에 정확히 붙으면(dock) 싣기·내리기를 자동 처리하는 것으로 단순화
- 출력: `{"plan": [{"skill": ..., ...}]}`. 없는 위치는 `{"plan": [], "error": "unknown_location"}`, 할 수 없는 일은 `"unsupported_command"` 로 거부 (엉뚱하게 움직이지 않게 하는 안전 장치)

## 학습 데이터 (make_data.py)
- 템플릿 기반으로 한국어 명령과 정답 JSON 쌍을 자동 생성 (시드 42 고정). 결과: data/train.jsonl 4,000 / val 300 / test 400
- 단일 작업(이동, 꺼내서 옮기기, 싣기, 내리기, 충전, 따라가기, 대기, 정지) + 두 작업 연결 + 거부 데이터(없는 위치, 할 수 없는 일)
- 한국어 조사 (으)로·을/를 를 끝 글자 받침에 맞춰 자동 선택. 숫자는 읽는 소리 기준 (1·7·8 은 ㄹ 받침이라 '로')
- 각 템플릿의 마지막 문장 패턴과 연결 표현 하나는 test 전용(held-out) → test 400개 중 200개는 학습 때 못 본 표현으로, 외운 문장이 아닌 일반화 능력을 측정
- 같은 문장이 train/val/test 사이에 겹치지 않게 함. 같은 split 안에서는 짧은 명령(정지 등)을 위해 같은 문장을 최대 3번까지 허용
- 두 작업 연결 시 follow 는 앞에 오지 않음 (계속 이어지는 동작이라 다음 작업으로 넘어갈 시점이 없음), 같은 작업 두 번 반복 제외
- 각 줄 필드: input, output(정답 JSON 문자열), template, heldout(처음 보는 패턴 여부)

## 평가 (evaluate.py)
- test 400개로 측정. 지표: JSON 유효율, 정확도(plan·error 완전 일치), 거부 정확도. '본 패턴'/'처음 보는 패턴' 으로 나눠 출력
- 파인튜닝 전 기준점과 파인튜닝 후 비교에 같은 스크립트·같은 시스템 프롬프트(schema.SYSTEM_PROMPT)를 씀
- 생성: greedy(do_sample=False), max_new_tokens 256, 배치 16, chat template 에 enable_thinking=False
- 문장별 예측은 results/<이름>.jsonl 에 저장

## 기준 측정 결과 (파인튜닝 전)
- Qwen/Qwen3.5-0.8B, test 50개(본 패턴 25 + 처음 보는 패턴 25), 2026-11-12: JSON 유효 70% (35/50), 정확도 0% (0/50), 거부 3개 중 0개
  - 주요 오류: 위치를 한국어 그대로 씀('B구역 3번 선반', 'C-4'), skill 이름에 괄호 포함('go_to(A1)'), dock·load·unload 누락, follow 남용, 같은 단계 무한 반복(토큰 한도까지), 거부해야 할 명령도 실행 계획을 만듦, 닫는 괄호 '}' 가 하나 더 붙어 JSON 파싱 실패
  - 위치를 정규화만 했으면 맞았을 사례도 있음 (예: 'D구역 4번까지 가 줄래?' → target 'D구역 4번')

## 학습 (train.py)
- Unsloth FastLanguageModel + LoRA(r=16, alpha=16, dropout 0, q/k/v/o/gate/up/down), bf16, 4bit 양자화 안 함
- 학습 루프는 TRL SFTTrainer 대신 transformers Trainer + DataCollatorForSeq2Seq 사용
  - 이유: Qwen3.5 는 config 에 비전 기능이 있어서 TRL SFTTrainer 가 VLM 으로 보고 assistant_only_loss 를 막는다는 사례가 있음 (Hugging Face 포럼). 프롬프트 부분 labels 를 -100 으로 직접 마스킹해서 답변(JSON)만 학습
- 프롬프트는 evaluate.build_prompt 를 그대로 사용 (학습·평가 형식 일치), 답변 끝에 <|im_end|>
- 기본값: epochs 2, lr 2e-4, cosine + warmup 5%(스텝 수로 계산해 warmup_steps 로 지정), batch 16, max_len 512
- 저장: <out>/lora (어댑터), <out>/merged (16bit 병합 모델, evaluate.py 로 평가)
- Unsloth 문서상 Qwen3.5 bf16 LoRA VRAM: 0.8B 약 3GB, 2B 약 5GB

## 학습 결과
### 1차: Qwen3.5-0.8B, 기본값 (2026-11-12)
- 500 스텝, 6.1분 (약 1.38 스텝/초), 학습 파라미터 6,389,760 / 전체 859,375,680 (0.74%), 최대 데이터 길이 318 토큰
- train loss 0.157(20 스텝) → 0.022(40) → 약 0.0001 이하(0.6 epoch 이후). eval_loss 0.0045(100 스텝) → 0.00058(200) → 0.00016(300~500)
- val 은 train 과 같은 문장 패턴이라 loss 가 매우 낮게 나오는 것이 정상. 일반화는 test 의 '처음 보는 패턴' 으로 판단해야 함
- 저장: outputs/qwen35_08b/lora, outputs/qwen35_08b/merged
- 무시해도 되는 경고: `causal_conv1d` 미설치로 PyTorch 기본 구현 사용(느리지만 결과는 같음), 종료 시 `nanobind: leaked ... instances` (triton 쪽 메시지, 학습·저장 결과에는 영향 없음)
- 평가 (test 400, 86초, 문장당 0.22초):
  | | JSON 유효 | 정확도 | 거부 정확도 |
  |---|---|---|---|
  | 전체 400 | 99.2% | 86.5% | 87.1% (31개) |
  | 본 패턴 200 | 100% | **100%** | - |
  | 처음 보는 패턴 200 | 98.5% | **73.0%** | 87.1% |
  - 파인튜닝 전 정확도 0% (50개 기준) 대비 크게 개선
- 오답 54개 분석: 거의 전부 처음 보는 패턴에서 발생
  - `{출발}에서 {도착}까지 {물건} 운반해` (꺼내서 옮기기의 held-out 패턴): 단독 22/32 오답 + 이 패턴이 들어간 두 작업 연결 대부분 오답. 출발지에서 dock·load 하지 않고 도착지에서 dock·load 한 뒤 출발지로 돌아가 unload — 'A에서 B까지' 구문의 출발·도착 역할을 뒤집음
  - `택배 {없는 위치}에 내려놔` (held-out 거부 패턴): G·H 구역 4개 오답 — 거부하지 않고 실행 계획을 만듦
  - `따라와 줘` (held-out follow 패턴): 연결 문장 끝의 follow 를 빠뜨림
  - 해석: 학습한 문장 패턴은 완벽히 처리하지만, 처음 보는 구문 구조는 일반화가 약함 → 문장 표현 다양성이 부족한 데이터 문제. 본 패턴 100% 만 봤다면 놓쳤을 약점을 held-out 평가가 드러냄

### 2차 계획: 데이터 v2 — 문장 표현 다양화 (2026-11-12)
- 변경: 작업별 학습용 문장 패턴을 늘림 (꺼내서 옮기기 5→14, 이동 5→11, 싣기 3→7, 내리기 3→7, 따라가기·대기·정지·거부·연결 표현도 추가). held-out 패턴(각 템플릿의 마지막 패턴, 연결 표현 '다 하면')은 그대로 test 전용
- test 는 1차와 같은 data/test.jsonl 을 그대로 사용 (make_data.py 가 기존 test 가 있으면 유지하고 train/val 만 재생성) → 1차와 공정 비교
- 학습 데이터 수(4,000)·학습 설정은 1차와 동일 → 표현 다양성 효과만 비교
- 주의: 추가한 패턴 중 `{출발}부터 {도착}까지 {물건} 옮겨` 는 held-out 패턴 `{출발}에서 {도착}까지 {물건} 운반해` 와 '까지' 구조가 비슷함. 실제 사용자도 쓰는 표현이라 넣었지만, 해당 held-out 패턴 점수 상승을 해석할 때 고려해야 함
- 결과 (2026-11-12): 학습 4.9분 (500 스텝), 평가 71초 (문장당 0.18초)
  | | 1차 (데이터 v1) | 2차 (데이터 v2) |
  |---|---|---|
  | 전체 정확도 | 86.5% | **97.8%** |
  | 본 패턴 | 100% | 100% |
  | 처음 보는 패턴 | 73.0% | **95.5%** |
  | JSON 유효 | 99.2% | 100% |
  | 거부 정확도 | 87.1% | **100%** (31개) |
  - 남은 오답 9개: 8개가 `{출발}에서 {도착}까지 {물건} 운반해` 패턴(이 패턴 69개 중 61개 정답, 1차는 대부분 오답) — 출발·도착을 뒤바꿔 C1 에서 싣고 C2 에 내리는 식. 나머지 1개는 `~ 실어 줘. 다 하면 ~까지 가 줄래?` 연결
  - 결론: 같은 모델·같은 데이터 수·같은 학습 설정에서 문장 표현 다양성만 늘려 처음 보는 패턴 정확도 73.0% → 95.5%. 소형 모델 파인튜닝에서는 데이터 표현 다양성이 일반화의 핵심
  - 해석 주의: '부터 ~까지 옮겨' 패턴을 추가한 영향으로 '에서 ~까지 운반해' 점수가 오른 부분이 있을 수 있음
### 3차: Qwen3.5-2B, 데이터 v2 (2026-11-12)
- 0.8B 2차와 같은 train/val/test·같은 학습 설정. 학습 파라미터 10,911,744 / 전체 2,224,153,408 (0.49%)
- 학습 8.0분 (500 스텝, 약 1.04 스텝/초), 평가 74초 (문장당 0.18초, 배치 16 기준이라 0.8B 와 거의 같음)
  | | 0.8B (데이터 v2) | 2B (데이터 v2) |
  |---|---|---|
  | 전체 정확도 | 97.8% (오답 9) | **99.0%** (오답 4) |
  | 본 패턴 | 100% | 99.5% |
  | 처음 보는 패턴 | 95.5% | **98.5%** |
  | 거부 정확도 | **100%** | 93.5% (31개 중 29) |
  | 학습 시간 | 4.9분 | 8.0분 |
- `{출발}에서 {도착}까지 {물건} 운반해` 패턴: 0.8B 61/69 → 2B 68/69
- 2B 오답 4개
  - `택배 정문에 내려놔` (×2, held-out 거부 패턴): 거부하지 않고 '정문'을 outbound(출고장)로 해석해 실행 계획 생성. 0.8B 는 둘 다 정확히 거부
    - 해석: 큰 모델은 사전 지식으로 없는 위치를 '가장 그럴듯한 위치'에 연결하려는 경향 → 로봇 안전 측면에서는 위험한 유형의 오답 (확인 필요: 같은 경향이 다른 없는 위치 표현에서도 나오는지)
  - 연결 문장에서 계획 끝에 불필요한 단계를 반복 생성 1건, 출발·도착 뒤바뀜 1건
- 결론: 모델 크기를 2.5배 키우면 처음 보는 표현의 일반화는 좋아지지만(95.5% → 98.5%), 거부 안전성은 오히려 떨어진 사례가 있음. 온디바이스 크기·학습 시간까지 고려하면 0.8B 도 충분히 실용적
- 다음 후보: 거부 데이터에 '실제로 있을 법한 장소 이름'(정문·사무실·하역장 등)을 더 넣어 2B 의 과잉 해석 줄이기 / 베이스 모델 확정 / 온디바이스 변환(GGUF) / Unity 강화학습 파트 시작

## 베이스 모델 후보
- Qwen/Qwen3.5-0.8B, Qwen/Qwen3.5-2B: Apache 2.0, Unsloth 공식 파인튜닝 지원. 모델 카드상 0.8B 는 '작업별 파인튜닝' 용도를 명시. 이미지 입력도 받는 모델이라 transformers 에서 CausalLM 클래스로 안 열릴 수 있음 (evaluate.py 는 ImageTextToText 클래스로 대체 시도)
- naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B: 한국어 특화, 텍스트 전용, 컨텍스트 16k. Hugging Face 에서 이용 조건 동의 후 받을 수 있음(gated). Unsloth 지원 여부 확인 필요

## 확정된 설계 결정
- 학습 방식: LoRA/QLoRA (GPU 메모리 12GB 기준)
- 학습 도구: Unsloth (공식 문서상 Windows 네이티브 지원, 실제 설치·동작은 확인 필요)
- 평가: 파인튜닝 전/후를 같은 테스트 세트로 비교 (JSON 형식 유효율, 명령 정확도)

## 미정 (확인 필요)
- 베이스 모델 확정 (기준 측정 후)
- 온디바이스 실행 방식 (예: GGUF + llama.cpp)

## 환경
- 학습·추론 기준: GPU 메모리 12GB
- OS: Windows 에서 진행 (ROS2 를 쓰지 않으므로 Linux/WSL 불필요). ML-Agents, Unsloth 모두 Windows 지원
- Python 환경은 두 개로 분리
  - LLM 파인튜닝용: conda 환경 `robot_llm` (2026-11-12 구축). ML-Agents 가 numpy·protobuf·Python 버전을 낮게 고정하므로 같은 환경에 두지 않음
    - Python 3.12, torch 2.14.1+cu130 (GPU 인식 확인), unsloth 2026.11.4, unsloth_zoo 2026.11.4, transformers 5.17.0, trl 1.15.0, peft 0.21.2, bitsandbytes 0.50.2, datasets 4.8.5, triton-windows 3.9.0.post30
    - 설치 순서: 환경 생성 → `pip install unsloth` → CUDA 빌드 torch 를 `--no-deps` 로 덮어쓰기 (unsloth 설치가 torch 를 CPU 판으로 바꾸므로)
  - 강화학습용: 기존 conda 환경 `mlagents` 재사용 (2026-11-12 확인)
    - Python 3.10.12, mlagents 1.1.0, mlagents_envs 1.1.0, torch 2.11.0+cu128, numpy 1.23.5, protobuf 3.20.3, onnx 1.22.0
- Unity ML-Agents 패키지: com.unity.ml-agents 4.0.3 (Python mlagents 1.1.0 과 함께 검증된 조합. 최신 4.1.0 과 1.1.0 의 호환은 확인 필요)
- Unity 프로젝트: unity/WarehouseRobot, 에디터 6000.3.17f1, com.unity.ml-agents 4.0.3 (2026-11-12 생성 확인)
  - 보안 권고: Unity 의 2026년 10월 보안 권고(인증기관 eMudhra 인증서 키 유출)에서 6000.3 LTS 는 6000.3.0a1~6000.3.27f1 이 영향 대상, 패치 버전은 6000.3.27f2. 에디터 쪽 위험은 Package Manager 가 위조된 패키지 레지스트리 응답을 받을 수 있는 것 등. ML-Agents 패키지 자체의 문제는 아님 → 같은 LTS 안의 패치 업데이트라 6000.3.27f2 로 올리기로 권장 (업데이트 후 ML-Agents 4.0.3 동작 확인 필요)
  - 참고: 2025년의 CVE-2025-59489(빌드된 앱의 인자 주입 취약점)는 6000.3.0b4 이후 버전에서 수정돼 6000.3.17f1 은 해당 없음
- Unity 폴더 구조 (2026-11-12 결정)
  ```
  Assets/
  ├─ 00Scene/
  ├─ 01Script/
  ├─ 02Resources/
  │   ├─ 00Component/   직접 만든 텍스처·모델·머티리얼 등 (Settings/: URP 설정 파일)
  │   └─ 01Asset/       에셋스토어에서 받은 것
  ├─ 03Image/
  ├─ 04Animation/
  └─ 05Audio/
  ```
  - 폴더 생성·이동은 Unity 에디터 안에서 처리 → .meta 와 GUID 유지
- 렌더 파이프라인: URP. 설정 파일은 Assets/02Resources/00Component/Settings (PC_RPAsset·PC_Renderer, Mobile_RPAsset·Mobile_Renderer, UniversalRenderPipelineGlobalSettings, DefaultVolumeProfile). Quality 의 Mobile → Mobile_RPAsset, PC → PC_RPAsset, Graphics 기본값 PC_RPAsset

## 시행착오 기록
### 2026-11-12 `pip install unsloth` 후 PyTorch 가 CPU 버전으로 바뀜
- 증상: cu128 PyTorch 를 먼저 설치하고 GPU 인식까지 확인한 뒤 `pip install unsloth` 를 했더니, `import unsloth` 에서 `torch.cuda.is_available() is False (torch 2.14.1+cpu)` 에러
- 원인: unsloth 가 더 높은 torch 버전(2.14.1)을 요구해서 pip 가 PyPI 에서 torch 를 다시 받음. Windows 용 PyPI torch 는 CPU 전용이라 GPU 버전이 덮어써짐
- 해결: `pip install --force-reinstall --no-deps torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130` → `2.14.1+cu130 True`, `import unsloth` 성공 (2026-11-12)
  - cu128 저장소에는 torch 2.11.0 까지만 있어서 2.14.1 을 받으려면 cu130 저장소가 필요 (드라이버가 CUDA 13 을 지원해야 함)
  - `--no-deps` 로 torch 만 교체해 unsloth 가 맞춰 둔 다른 패키지 버전은 유지. torchvision 0.29.2 는 cu130 저장소에 없어서 함께 지정하면 명령 전체가 실패함 → torch 만 설치
  - 남은 경고: 설치된 xformers 0.0.35 가 torch 2.10.0+cu128 용으로 빌드돼 로드되지 않음 → Unsloth 가 PyTorch 기본 attention 으로 대신 동작 (학습은 가능)

### 2026-11-12 TrainingArguments 에 warmup_ratio 인자가 없음
- 증상: `TypeError: TrainingArguments.__init__() got an unexpected keyword argument 'warmup_ratio'`
- 원인: 설치된 transformers 5.17.0 의 TrainingArguments 에는 warmup_ratio 가 없음 (인터넷 예제는 예전 버전 기준인 경우가 많음)
- 해결: 전체 스텝 수의 5% 를 계산해서 warmup_steps 로 지정 → 학습 정상 진행 확인 (2026-11-12)

### 2026-11-12 템플릿 파일 정리 중 Settings 폴더를 지워 렌더 파이프라인이 Built-in 으로 바뀜
- 증상: Assets 정리 후 렌더 파이프라인 확인 결과가 'Built-in'. Assets 루트에 UniversalRenderPipelineGlobalSettings.asset, DefaultVolumeProfile.asset 이 자동으로 다시 생김
- 원인: URP 템플릿의 Assets/Settings 폴더에 URP 파이프라인 에셋(PC_RPAsset 등)이 있었는데 함께 삭제됨 → Graphics·Quality 설정의 파이프라인 참조가 사라져 Built-in 으로 동작
- 해결: 에디터 스크립트로 UniversalRendererData + UniversalRenderPipelineAsset 을 PC·Mobile 용으로 새로 만들고 GraphicsSettings.defaultRenderPipeline 과 Quality 레벨별 renderPipeline 에 지정 → 'Universal (URP)' 확인. 자동 재생성된 두 파일은 Settings 폴더로 이동
- 교훈: URP 프로젝트에서 Settings 폴더는 지우면 안 되는 설정 파일 묶음. 정리할 때는 TutorialInfo·Readme 같은 안내 파일만 지울 것
