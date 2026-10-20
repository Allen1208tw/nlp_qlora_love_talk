"""
chat.py

載入 Qwen2.5-3B-Instruct + 最佳 LoRA adapter，
在 terminal 和「土味情話專家」進行多輪互動對話。

指令：

    /reset    清除對話歷史，重新開始（保留 system prompt）
    /history  顯示目前的對話歷史
    /help     顯示指令
    exit / quit / q   結束

多輪對話的原理：每一輪都把「system + 完整歷史 + 新問題」一起送進模型，
所以模型可以理解「那如果她問…」這種承接上一輪的追問。
"""

import torch

from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

from prepare_chat_data import SYSTEM_PROMPT


ADAPTER_DIR = "outputs/love_talk_lora"

MAX_NEW_TOKENS = 160
TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.1

# 最多保留最近 10 輪 user / assistant，避免 context 無限增長。
MAX_HISTORY_TURNS = 10


def load_model():
    """載入 tokenizer 與 base model + LoRA adapter。"""

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16 if torch.cuda.is_available() else torch.float32
    )

    model = AutoPeftModelForCausalLM.from_pretrained(
        ADAPTER_DIR,
        dtype=dtype,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def trim_history(messages):
    """保留 system prompt 與最近 MAX_HISTORY_TURNS 輪對話。"""

    if len(messages) <= 1 + MAX_HISTORY_TURNS * 2:
        return messages
    return [messages[0]] + messages[-MAX_HISTORY_TURNS * 2:]


def generate_reply(model, tokenizer, messages):
    """把完整對話歷史送進模型，只回傳這一輪新產生的 assistant 文字。"""

    inputs = tokenizer.apply_chat_template(
        trim_history(messages),
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    input_length = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(
        output[0][input_length:],
        skip_special_tokens=True,
    ).strip()


def new_conversation():
    return [{"role": "system", "content": SYSTEM_PROMPT}]


HELP_TEXT = "指令：/reset 重新開始、/history 顯示歷史、/help 說明、exit 離開"


def main():

    print("正在載入 Base Model + LoRA Adapter...")
    model, tokenizer = load_model()

    messages = new_conversation()

    print("=" * 60)
    print("土味情話專家上線囉！有什麼想撩的對象都可以問我～")
    print(HELP_TEXT)
    print("=" * 60)

    while True:
        try:
            user_input = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "q"):
            break

        if user_input == "/reset":
            messages = new_conversation()
            print("（已清除對話歷史）")
            continue

        if user_input == "/history":
            for m in messages[1:]:
                speaker = "你" if m["role"] == "user" else "專家"
                print(f"{speaker}：{m['content']}")
            continue

        if user_input == "/help":
            print(HELP_TEXT)
            continue

        messages.append({"role": "user", "content": user_input})
        reply = generate_reply(model, tokenizer, messages)
        messages.append({"role": "assistant", "content": reply})

        print(f"專家：{reply}")

    print("掰掰，祝你脫單成功！")


if __name__ == "__main__":
    main()
