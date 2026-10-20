"""
train_qlora.py

使用 4-bit QLoRA 微調 Qwen2.5-3B-Instruct，
建立可以多輪對話的「土味情話專家」。

修改自課堂範例 cases/llm_finetune/train_qlora.py，主要差異：

1. 模型由 Qwen2.5-0.5B-Instruct 換成 Qwen2.5-3B-Instruct。
   小模型的中文幽默與雙關能力太弱；3B 在 4-bit 下仍可在 8 GB VRAM 訓練。

2. 訓練環境是 NVIDIA RTX 4060（Ada 架構，compute capability 8.9），
   原生支援 BF16，因此改用：

       4-bit NF4 base model + BF16 compute + BF16 mixed precision

   不需要範例中為了 Tesla T4 把 LoRA 參數轉回 FP32 的處理。

3. 語料只有約 120 段對話（約 450 筆 examples），資料量小：
   - effective batch size 改為 16（4 × 4）
   - 每 10 個 optimizer steps 做一次 validation
   - LoRA rank 提高到 16，learning rate 2e-4

流程：

    data/sft/train.jsonl / valid.jsonl
        ↓
    Qwen2.5-3B-Instruct（4-bit NF4）
        ↓
    LoRA adapter（只訓練約 1% 參數）
        ↓
    SFTTrainer（只對 assistant completion 計算 loss）
        ↓
    validation + early stopping
        ↓
    outputs/love_talk_lora/  ← 最佳 LoRA adapter
"""

from pathlib import Path

import torch

from datasets import load_dataset

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)

from trl import SFTConfig, SFTTrainer


# =============================================================================
# 1. 模型與路徑設定
# =============================================================================

# 實驗過 Qwen2.5-1.5B-Instruct（結果在 experiments/qwen25_1.5b/），
# 風格學得到但梗常不通順，因此最終改用 3B。
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

DATA_DIR = Path("data/sft")
TRAIN_FILE = DATA_DIR / "train.jsonl"
VALID_FILE = DATA_DIR / "valid.jsonl"

# Trainer 的 checkpoint-* 放這裡（不上傳 GitHub）。
CHECKPOINT_DIR = Path("checkpoints")

# 最佳 LoRA adapter 放這裡（上傳 GitHub）。
OUTPUT_DIR = Path("outputs/love_talk_lora")


# =============================================================================
# 2. 訓練設定
# =============================================================================

# 語料都是短對話，512 tokens 足以放下 system + 4 輪上下文。
MAX_LENGTH = 512

EPOCHS = 8
LEARNING_RATE = 2e-4

# effective batch size = 4 × 4 = 16
PER_DEVICE_TRAIN_BATCH_SIZE = 4
PER_DEVICE_EVAL_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4

# 前 10 個 optimizer steps 做 linear warmup。
WARMUP_STEPS = 10

GRADIENT_CHECKPOINTING = True
OPTIM = "paged_adamw_8bit"

LOGGING_STEPS = 5
CHECK_STEPS = 10
CHECKPOINT_LIMIT = 2
EARLY_STOPPING_PATIENCE = 3

# 只對 assistant 回答的部分計算 loss，不要求模型背 user 的問題。
COMPLETION_ONLY_LOSS = True

# Qwen chat template 的訊息結束 token。
EOS_TOKEN = "<|im_end|>"

SEED = 42


# =============================================================================
# 3. LoRA 設定
# =============================================================================

LORA_R = 16
LORA_ALPHA = 32          # scaling = alpha / r = 2
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = "all-linear"


# =============================================================================
# 4. 輔助函式
# =============================================================================

def print_gpu_info():
    if not torch.cuda.is_available():
        raise RuntimeError("找不到 CUDA GPU，請先確認 NVIDIA 驅動與 PyTorch CUDA 版本。")

    print("=" * 80)
    print(f"GPU：{torch.cuda.get_device_name(0)}")
    print(f"CUDA capability：{torch.cuda.get_device_capability(0)}")
    print(f"PyTorch CUDA：{torch.version.cuda}")
    print(f"BF16 native support：{torch.cuda.is_bf16_supported(including_emulation=False)}")
    print("=" * 80)


# =============================================================================
# 5. 主流程
# =============================================================================

def main():

    print_gpu_info()

    # BF16 需要 Ampere（8.x）以後的 GPU；舊 GPU（例如 T4）自動退回 FP16。
    use_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

    for path in (TRAIN_FILE, VALID_FILE):
        if not path.is_file():
            raise FileNotFoundError(f"找不到 {path}，請先執行 python prepare_chat_data.py")

    # -------------------------------------------------------------------------
    # Step 1：載入 dataset（prompt / completion 格式）
    # -------------------------------------------------------------------------

    data = load_dataset(
        "json",
        data_files={
            "train": str(TRAIN_FILE),
            "validation": str(VALID_FILE),
        },
    )
    print(data)

    # -------------------------------------------------------------------------
    # Step 2：Tokenizer
    # -------------------------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -------------------------------------------------------------------------
    # Step 3：4-bit NF4 量化載入 base model
    # -------------------------------------------------------------------------

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quant_config,
        dtype=compute_dtype,
        device_map={"": 0},
    )

    # 訓練時不需要 KV cache（且會和 gradient checkpointing 衝突）。
    model.config.use_cache = False

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=GRADIENT_CHECKPOINTING,
    )

    # -------------------------------------------------------------------------
    # Step 4：加上 LoRA adapter
    # -------------------------------------------------------------------------

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # -------------------------------------------------------------------------
    # Step 5：SFT 訓練設定
    # -------------------------------------------------------------------------

    args = SFTConfig(
        output_dir=str(CHECKPOINT_DIR),
        max_length=MAX_LENGTH,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        bf16=use_bf16,
        fp16=not use_bf16,
        optim=OPTIM,
        logging_steps=LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=CHECK_STEPS,
        save_strategy="steps",
        save_steps=CHECK_STEPS,
        save_total_limit=CHECKPOINT_LIMIT,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        completion_only_loss=COMPLETION_ONLY_LOSS,
        eos_token=EOS_TOKEN,
        report_to="none",
        seed=SEED,
    )

    # -------------------------------------------------------------------------
    # Step 6：SFTTrainer + Early Stopping
    # -------------------------------------------------------------------------

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        processing_class=tokenizer,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=EARLY_STOPPING_PATIENCE
            )
        ],
    )

    # -------------------------------------------------------------------------
    # Step 7：開始訓練
    # -------------------------------------------------------------------------

    trainer.train()

    print(f"最佳 checkpoint：{trainer.state.best_model_checkpoint}")
    print(f"最佳 eval_loss：{trainer.state.best_metric}")

    # -------------------------------------------------------------------------
    # Step 8：儲存最佳 LoRA adapter、tokenizer 與 trainer state
    # -------------------------------------------------------------------------
    #
    # load_best_model_at_end=True，此時 trainer.model 已是最佳 checkpoint。
    # PEFT 模型 save_model() 只會存 LoRA adapter（幾十 MB），不會複製整個 base model。

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    trainer.state.save_to_json(str(OUTPUT_DIR / "trainer_state.json"))

    print(f"最佳 LoRA adapter：{OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
