"""
demo_chat.py

用預先寫好的多輪問題自動和模型對話，並把結果存成 results/demo_chat.md，
方便放進 README 展示。

每一組問題會跑兩次：

    1. 微調前：原始 base model（停用 LoRA adapter）
    2. 微調後：base model + 土味情話 LoRA adapter

兩者使用同一個 system prompt，可以直接比較微調的效果。

注意：這些問題都「不在訓練語料中」，用來測試模型的泛化能力。
"""

from pathlib import Path

from transformers import set_seed

from chat import generate_reply, load_model, new_conversation


OUTPUT_FILE = Path("results/demo_chat.md")

SEED = 42

# 每一組是一段多輪對話中，使用者依序說的話。
DEMO_DIALOGUES = [
    [
        "我暗戀的學妹在學校咖啡廳打工，我要怎麼跟她搭話？",
        "她回我「你要點什麼？」",
        "那她如果說「這不在菜單上」呢？",
    ],
    [
        "女朋友說我都不陪她，生氣了，該怎麼哄？",
        "她說「你每次都只會說好聽話」",
        "她終於笑了，接下來要說什麼？",
    ],
    [
        "我是資工系的，教我一句跟寫程式有關的情話。",
        "她不懂程式，聽不懂怎麼辦？",
        "再來一句比較不肉麻的。",
    ],
]


def run_dialogue(model, tokenizer, questions):
    messages = new_conversation()
    for question in questions:
        messages.append({"role": "user", "content": question})
        reply = generate_reply(model, tokenizer, messages)
        messages.append({"role": "assistant", "content": reply})
    return messages[1:]


def format_dialogue(messages):
    lines = []
    for m in messages:
        speaker = "你" if m["role"] == "user" else "專家"
        lines.append(f"{speaker}：{m['content']}")
    return "\n".join(lines)


def main():

    print("正在載入 Base Model + LoRA Adapter...")
    model, tokenizer = load_model()

    # 從 adapter_config.json 讀出 base model 名稱，例如 Qwen2.5-3B-Instruct。
    base_name = model.peft_config["default"].base_model_name_or_path.split("/")[-1]

    sections = ["# 多輪對話示範：微調前 vs 微調後\n"]

    for i, questions in enumerate(DEMO_DIALOGUES, start=1):

        # 微調前：暫時停用 LoRA adapter，等同原始 base model。
        set_seed(SEED)
        with model.disable_adapter():
            before = run_dialogue(model, tokenizer, questions)

        # 微調後：啟用 LoRA adapter。
        set_seed(SEED)
        after = run_dialogue(model, tokenizer, questions)

        block = (
            f"## 對話 {i}\n\n"
            f"### 微調前（原始 {base_name}）\n\n"
            f"```\n{format_dialogue(before)}\n```\n\n"
            f"### 微調後（+ 土味情話 LoRA）\n\n"
            f"```\n{format_dialogue(after)}\n```\n"
        )
        sections.append(block)
        print(block)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text("\n".join(sections), encoding="utf-8")
    print(f"已輸出：{OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
