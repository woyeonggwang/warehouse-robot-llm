# warehouse-robot-llm

한국어 자연어 명령을 로봇이 실행할 **기술(skill) 순서 JSON** 으로 바꾸는 소형 LLM 과, 각 기술을 실행하는 **강화학습 에이전트(Unity ML-Agents)** 를 잇는 프로젝트입니다. 첫 시나리오는 실내 창고의 바퀴형 운반 로봇이고, 최종 목표는 같은 구조(소형 LLM + 강화학습 제어)를 휴머노이드에 적용해 보는 것입니다.

```
"B3 상자 꺼내서 포장대로 가져다줘"
      │  소형 LLM (Qwen3.5-2B + LoRA)
      ▼
{"plan": [{"skill": "go_to", "target": "B3"}, {"skill": "dock", "target": "B3"},
          {"skill": "load", "item": "box"}, {"skill": "go_to", "target": "packing"}, {"skill": "unload"}]}
      │  장소 점수 모델 → 실행 / 확인 질문 / 거부
      ▼
Unity 강화학습 에이전트가 go_to · dock · follow 를 실행 (load · unload 등은 스크립트로 처리)
```

| 파트 | 상태 |
|---|---|
| LLM: 명령 체계, 데이터 생성, LoRA 파인튜닝, 평가, 과잉 해석 방지 | **완료** (2026-11-13) |
| 강화학습: Unity 창고 환경, go_to → dock → follow 에이전트 | 진행 중 |
| 연결: LLM 출력 → 에이전트 실행 | 예정 (`llm/infer.py` HTTP 서버로 연결) |

## 명령 체계

기준 파일은 [llm/schema.py](llm/schema.py) 입니다. 데이터 생성·학습·평가가 모두 이 파일을 참조합니다.

- 위치: 선반 A1~D4 (16칸), `inbound` (입고장), `outbound` (출고장), `packing` (포장대), `charger` (충전소)
- 물건: `box` (상자), `parcel` (택배), `tote` (바구니)
- 기술
  - 강화학습 담당: `go_to(target)`, `dock(target)`, `follow(target: me|worker)`
  - 스크립트로 즉시 처리: `load(item)`, `unload`, `charge`, `wait(sec)`, `stop`
- 거부: 목록에 없는 위치는 `{"plan": [], "error": "unknown_location"}`, 할 수 없는 일은 `"unsupported_command"`. 엉뚱한 곳으로 움직이지 않게 하는 안전 장치입니다.

바퀴 로봇이 물건을 직접 집는 대신, 선반에 정확히 붙으면(`dock`) 싣기·내리기를 자동으로 처리하는 것으로 단순화했습니다.

## LLM 파트

### 데이터 ([make_data.py](llm/make_data.py))

- 템플릿으로 한국어 명령과 정답 JSON 쌍을 만듭니다 (시드 42). train 4,000 / val 300 / test 400
- 한 작업(이동, 꺼내서 옮기기, 싣기, 내리기, 충전, 따라가기, 대기, 정지), 두 작업 연결, 거부 데이터를 포함합니다
- 조사 `(으)로`·`을/를` 는 끝 글자 받침에 맞춰 고릅니다 (숫자는 읽는 소리 기준)
- **처음 보는 표현 평가**: 각 템플릿의 마지막 문장 패턴과 연결 표현 하나는 test 에만 씁니다. test 400개 중 200개가 학습 때 못 본 표현이라, 외운 문장이 아니라 일반화 능력을 잽니다

### 학습 ([train.py](llm/train.py))

- Unsloth + LoRA (r 16, alpha 16, q/k/v/o/gate/up/down), bf16. epochs 2, lr 2e-4, batch 16
- 학습 루프는 transformers `Trainer` 를 쓰고, 프롬프트 부분의 labels 를 -100 으로 가려 답변 JSON 만 학습합니다
  - TRL `SFTTrainer` 는 Qwen3.5 를 비전 모델로 보고 답변만 학습하는 옵션을 막는다는 사례가 있어서 쓰지 않았습니다
- 0.8B 약 5~6분, 2B 약 8~9분 (500 스텝)

### 결과 (test 400개, greedy 생성)

| 모델 · 데이터 | 전체 정확도 | 본 패턴 | 처음 보는 패턴 | 거부 정확도 (31개) |
|---|---|---|---|---|
| Qwen3.5-0.8B 파인튜닝 전 (50개만) | 0% | - | - | 0/3 |
| 0.8B · 데이터 v1 | 86.5% | 100% | 73.0% | 87.1% |
| 0.8B · 데이터 v2 (문장 표현 다양화) | 97.8% | 100% | 95.5% | 100% |
| 2B · 데이터 v2 | 99.0% | 99.5% | 98.5% | 93.5% |
| **2B · 데이터 v2 · strict 프롬프트 (최종)** | **99.5%** | **100%** | **99.0%** | **100%** |

정확도는 plan·error 가 정답 JSON 과 완전히 같은 비율입니다. 파인튜닝한 모델은 모두 JSON 유효율이 99% 이상입니다.

- 모델 크기·데이터 수·학습 설정을 그대로 두고 **문장 표현만 다양하게 늘려** 처음 보는 패턴 정확도가 73.0% → 95.5% 로 올랐습니다. 소형 모델에서는 데이터 표현 다양성이 일반화에 가장 크게 작용했습니다
- 본 패턴만 봤다면 100% 로 보였을 약점을 held-out 평가가 드러냈습니다 (예: "A에서 B까지 운반해" 의 출발·도착을 뒤집음)

### 과잉 해석 방지: 목록에 없는 장소를 어떻게 다룰까

test 의 거부 문장은 대부분 "G3" 같은 코드형이라 거부율이 높게 나왔습니다. 그래서 '하역장·경비실·후문' 처럼 있을 법한 장소로 따로 점검 세트(64개)를 만들었더니, 모델이 절반가량을 거부하지 않고 비슷한 공식 위치로 바꿔 실행했습니다 (하역장 → inbound 등). 0.8B 는 54.7%, 2B 는 51.6% 만 거부했고, strict 프롬프트로 학습해도 75.0% 였습니다.

하지만 '하역장 → 입고장' 은 사용자 의도에 맞는 해석일 수도 있어서, 모두 거부하는 것이 정답이라고 보기 어렵습니다. 그래서 거부 여부를 바로 정하지 않고, 장소마다 **점수 두 개**를 매겨 행동을 정하도록 바꿨습니다.

- **창고 업무 관계율** (0~1): 그 장소가 창고 물류 작업과 얼마나 관련 있는가
- **위험률** (0~1): 로봇이 그 장소로 잘못 갔을 때의 피해 크기

라벨링과 점수 모델

- 정답 점수는 로컬 LLM (Ollama `qwen3.5:9b`) 이 초안을 매기고, 사람이 정한 검토 기준으로 고쳤습니다. 9B 라벨러도 '목록 밖 물류 장소에 관계율 1.0' 처럼 같은 과잉 해석을 보여서, 검토 기준을 스크립트(`fix_labels*.py`)로 남겼습니다
- 평가용 106곳 ([data/labels.csv](llm/data/labels.csv)) 은 고정하고, 보강용 133곳 (위험 장소 75, 공식 위치 별칭·경계 장소 58) 은 학습에만 씁니다
- 점수 모델 ([train_scorer.py](llm/train_scorer.py)): 해석 모델은 그대로 두고, 명령 문장의 내부 표현(9~21층 마지막 토큰) 위에 릿지 회귀를 학습합니다. 숫자를 텍스트로 생성하게 하는 것보다 정밀하고, 해석 모델을 다시 학습할 필요가 없습니다
- 평가는 장소 단위 중첩 교차검증입니다 (시험 장소의 문장은 학습·설정 선택에 쓰지 않음). 평가 106곳 기준 MAE 관계율 0.091 · 위험률 0.094, 장소 단위 행동 일치 95.3%

행동 규칙

- 할 수 없는 일 → 거부. 정지·대기·따라가기만 있는 계획 → 실행
- 그 밖: **관계율 ≥ 0.9 이고 위험률 < 0.4 → 실행 / 관계율 ≥ 0.5 → 확인 질문 / 그 밖 → 거부**
- 기준값은 학습용 데이터로만 정했습니다. 관계율 기준은 라벨 구간의 의미 그대로이고, 위험률 상한은 공식 위치 명령의 95% 가 통과하는 값입니다

해석 모델 + 점수 모델 연결 점검 ([e2e_check.py](llm/e2e_check.py))

| | 해석 모델만 | 해석 + 점수 모델 |
|---|---|---|
| 처음 보는 장소 848문장 중 잘못 실행 (위험률 0.5 이상인 곳) | 123 (28) | **2 (0)** |
| test 정상 명령 바르게 실행 | 367/369 | 309/369 (83.7%) |
| test 없는 위치 명령 실행 | 0/19 | 0/19 |
| test 할 수 없는 일 거부 | 12/12 | 12/12 |

목록에 없는 장소로 잘못 실행하는 경우가 123문장에서 2문장으로 줄었습니다. 그 대가로 정상 명령의 16% 는 바로 실행하지 않고 "입고장으로 보면 될까요?" 처럼 되묻습니다. 거부로 막힌 정상 명령은 없습니다.

## 실행 방법

### 환경

- conda 환경 두 개를 씁니다. ML-Agents 가 numpy·protobuf 버전을 낮게 고정해서 분리했습니다
  - `robot_llm` (LLM): Python 3.12, torch 2.14.1+cu130, unsloth 2026.11.4, transformers 5.17.0
  - `mlagents` (강화학습): Python 3.10, mlagents 1.1.0, Unity 6000.3 + com.unity.ml-agents 4.0.3

```bash
conda create -n robot_llm python=3.12 && conda activate robot_llm
pip install unsloth
# Windows 에서는 unsloth 설치가 torch 를 CPU 판으로 바꾸므로 CUDA 판으로 다시 덮어씀
pip install --force-reinstall --no-deps torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
```

### 학습·평가 (`llm/` 에서)

```bash
python make_data.py                     # data/train·val.jsonl (test.jsonl 은 고정, 이미 있으면 유지)
python train.py --model unsloth/Qwen3.5-2B --out outputs/qwen35_2b_v2_strict --prompt strict
python evaluate.py --model outputs/qwen35_2b_v2_strict/merged --prompt strict --name ft_qwen35_2b_v2_trained_strict
```

점수 모델과 기준값 다시 만들기

```bash
python make_scorer_sentences.py         # data/scorer_sentences.csv
python train_scorer.py --extra data/labels_add.csv,data/labels_alias.csv --sentences data/scorer_sentences.csv --name scorer_2b_strict_sent --repeats 2
python tune_thresholds.py --repeats 1   # results/thresholds.json
python e2e_check.py                     # 연결 점검
```

모델 가중치(`llm/outputs/`)는 용량이 커서 저장소에 올리지 않았습니다. 위 명령으로 다시 만들 수 있습니다.

### 명령 해석 서버 ([infer.py](llm/infer.py))

```bash
python infer.py "하역장으로 가"            # 한 문장
python infer.py --serve --port 8765        # 127.0.0.1 HTTP 서버 (Unity 연결용)
```

`POST /command` 에 `{"text": "하역장으로 가"}` 를 보내면 다음과 같은 응답이 옵니다 (점수는 예시).

```json
{"input": "하역장으로 가", "action": "확인",
 "plan": [{"skill": "go_to", "target": "inbound"}], "error": null,
 "relevance": 0.85, "risk": 0.35, "reason": "점수 확인",
 "message": "입고장으로 보면 될까요?", "latency_ms": 450}
```

`action` 이 `확인` 일 때 `plan` 은 사용자가 '네' 하면 실행할 계획입니다. 처리 시간은 한 문장에 0.4~2.7초입니다 (2B bf16, 두 작업 연결이 가장 김).

## 강화학습 파트 (진행 중)

- 첫 기술은 `go_to`, 이후 `dock` (정밀 정차) → `follow`
- 바퀴 토크 제어 (좌우 바퀴 토크 연속 행동 2개). 속도 제어보다 학습은 어렵지만 실제 로봇에 가깝고, 토크로 제어하는 휴머노이드로 이어집니다
- 구조: 경로 계획기가 전체 경로를 정하고, 강화학습 에이전트가 그 경로를 따라가며 장애물을 피합니다. 실제 창고 로봇(위치 파악 + 경로 계획 + 현장 제어)과 같은 나눔이고, 휴머노이드에서도 '어디로'(계획기)와 '어떻게 걸을지'(강화학습)로 같은 구조를 씁니다
- NavMesh 길찾기와 도착률·시간·충돌을 비교할 예정입니다

## 저장소 구조

```
warehouse-robot-llm/
├─ llm/
│  ├─ schema.py               명령 체계·시스템 프롬프트 (기준 파일)
│  ├─ make_data.py            학습·평가 데이터 생성
│  ├─ train.py / evaluate.py  LoRA 파인튜닝 / 평가
│  ├─ make_probe.py           없는 장소 점검 세트
│  ├─ label_with_llm.py, fix_labels*.py      장소 점수 라벨 (9B 초안 → 검토 기준 적용)
│  ├─ make_scorer_sentences.py, train_scorer.py, tune_thresholds.py   점수 모델·기준값
│  ├─ e2e_check.py            해석 + 점수 모델 연결 점검
│  ├─ infer.py                최종 진입점 (CLI · HTTP)
│  ├─ data/                   학습·평가 데이터, 장소 라벨
│  └─ results/                평가 결과 (문장별 예측 포함)
├─ unity/WarehouseRobot/      Unity 프로젝트 (URP)
└─ rl/configs/                ML-Agents 학습 설정
```

## 사용한 모델

- 베이스 모델: [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) (Apache 2.0). 비교 실험에 Qwen3.5-0.8B
- 라벨 초안: Ollama `qwen3.5:9b`
