"""Unsloth 로 LoRA 파인튜닝.

실행 예:
  python train.py                                   # 기본: Qwen3.5-0.8B
  python train.py --model unsloth/Qwen3.5-2B --out outputs/qwen35_2b
결과:
  <out>/lora    : LoRA 어댑터만 (수 MB)
  <out>/merged  : 베이스에 LoRA 를 합친 16bit 모델 → evaluate.py 로 바로 평가 가능
평가:
  python evaluate.py --model outputs/qwen35_08b/merged --name ft_qwen35_08b
"""
import argparse
import json
import math
import time

from unsloth import FastLanguageModel        # unsloth 는 transformers 보다 먼저 import 해야 최적화 패치가 적용됨
import torch
from datasets import Dataset
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

from evaluate import build_prompt            # 평가 때와 완전히 같은 프롬프트 형식으로 학습


def load_rows(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Qwen3.5-0.8B")
    ap.add_argument("--out", default="outputs/qwen35_08b")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--r", type=int, default=16, help="LoRA rank")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=512)
    args = ap.parse_args()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model, max_seq_length=args.max_len, dtype=torch.bfloat16, load_in_4bit=False)
    tok = getattr(tokenizer, "tokenizer", tokenizer)   # 이미지도 받는 모델은 processor 가 반환될 수 있어서 안쪽 tokenizer 사용

    model = FastLanguageModel.get_peft_model(
        model, r=args.r, lora_alpha=args.r, lora_dropout=0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=42)
    model.print_trainable_parameters()

    # 답변 끝 표시 토큰: Qwen 계열은 <|im_end|>, 그 외는 eos
    end = "<|im_end|>" if "<|im_end|>" in tok.get_vocab() else tok.eos_token

    def encode(row):
        prompt_ids = tok(build_prompt(tok, row["input"]), add_special_tokens=False)["input_ids"]
        answer_ids = tok(row["output"] + end, add_special_tokens=False)["input_ids"]
        ids = (prompt_ids + answer_ids)[:args.max_len]
        labels = ([-100] * len(prompt_ids) + answer_ids)[:args.max_len]
        # labels 에서 프롬프트 부분은 -100 → loss 계산에서 제외. 모델은 '답변(JSON)' 만 배움
        return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}

    train = Dataset.from_list(load_rows("data/train.jsonl")).map(encode, remove_columns=["input", "output", "template", "heldout"])
    val = Dataset.from_list(load_rows("data/val.jsonl")).map(encode, remove_columns=["input", "output", "template", "heldout"])
    print(f"train {len(train)}, val {len(val)}, 최대 길이 {max(len(x) for x in train['input_ids'])} 토큰")

    total_steps = math.ceil(len(train) / args.batch) * args.epochs
    warmup_steps = int(total_steps * 0.05)              # 전체 스텝의 5% 동안 warmup
    print(f"총 {int(total_steps)} 스텝, warmup {warmup_steps} 스텝")

    trainer = Trainer(
        model=model,
        train_dataset=train,
        eval_dataset=val,
        data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
        args=TrainingArguments(
            output_dir=f"{args.out}/checkpoints",
            per_device_train_batch_size=args.batch,
            per_device_eval_batch_size=args.batch,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",          # warmup + cosine
            warmup_steps=warmup_steps,           # transformers 5.17 에는 warmup_ratio 인자가 없어서 스텝 수로 직접 지정
            logging_steps=20,
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="no",
            bf16=True,
            optim="adamw_torch",
            report_to="none",
            seed=42,
        ),
    )
    t0 = time.time()
    trainer.train()
    print(f"학습 시간: {(time.time() - t0) / 60:.1f}분")

    model.save_pretrained(f"{args.out}/lora")
    tok.save_pretrained(f"{args.out}/lora")
    model.save_pretrained_merged(f"{args.out}/merged", tokenizer, save_method="merged_16bit")
    print(f"저장 완료: {args.out}/lora, {args.out}/merged")


if __name__ == "__main__":
    main()
