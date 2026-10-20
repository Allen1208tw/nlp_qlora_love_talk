"""
prepare_chat_data.py

把 data/raw/*.jsonl 的多輪對話語料整理成 TRL SFTTrainer 使用的
prompt / completion 格式。

流程（沿用課堂範例 llm_finetune 的設計）：

    data/raw/part_*.jsonl      （每行一段完整多輪對話）
        ↓ 合併、清理、去除重複
    data/chat.jsonl            （加上 system prompt 的完整對話）
        ↓ 以「整段對話」為單位切 train / validation
        ↓ 再各自展開成 prompt / completion
    data/sft/train.jsonl
    data/sft/valid.jsonl

為什麼要「先切再展開」？

    同一段對話展開後的每一筆 example 都共用前面的上下文。
    如果先展開再切，validation 可能包含 train 看過的同一段對話，
    造成 turn-level leakage，eval_loss 會被低估。
"""

import json
import random
from pathlib import Path


# =============================================================================
# 1. 設定
# =============================================================================

RAW_DIR = Path("data/raw")
CHAT_FILE = Path("data/chat.jsonl")
OUTPUT_DIR = Path("data/sft")

# 「土味情話專家」的角色設定，訓練與推論必須使用同一段 system prompt。
SYSTEM_PROMPT = (
    "你是一位土味情話高手，會以親切、溫和又幽默的口吻回答使用者的問題。"
    "請使用繁體中文，教使用者一句可以直接說出口的情話，並用「」框起來。"
    "回答要簡短有梗、尊重對方，不低俗、不油膩過頭。"
)

VALID_RATIO = 0.1

# prompt 最多保留最近幾則 user / assistant messages（system 固定保留）。
#
# 8 則 ≈ 最近 4 輪，語料每段 3～5 輪，幾乎可以保留完整上下文。
MAX_CONTEXT_MESSAGES = 8

SEED = 42


# =============================================================================
# 2. 輔助函式
# =============================================================================

def load_raw_conversations():
    """
    讀取 data/raw/ 下所有 .jsonl，回傳 list[list[message]]。

    同時做基本清理：
    - 去除前後空白
    - 檢查 user / assistant 必須交替，且以 assistant 結尾
    - 以「第一句 user」去除重複對話
    """

    conversations = []
    seen = set()

    for path in sorted(RAW_DIR.glob("*.jsonl")):
        count = 0
        with open(path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue

                messages = [
                    {"role": m["role"], "content": m["content"].strip()}
                    for m in json.loads(line)["messages"]
                    if m["role"] in ("user", "assistant")
                ]

                valid = (
                    len(messages) >= 2
                    and len(messages) % 2 == 0
                    and all(
                        m["role"] == ("user" if i % 2 == 0 else "assistant")
                        and m["content"]
                        for i, m in enumerate(messages)
                    )
                )
                if not valid:
                    print(f"  略過格式錯誤：{path.name} 第 {line_no} 行")
                    continue

                key = messages[0]["content"]
                if key in seen:
                    print(f"  略過重複對話：{path.name} 第 {line_no} 行")
                    continue
                seen.add(key)

                conversations.append(messages)
                count += 1

        print(f"{path.name}：{count} 段對話")

    return conversations


def build_examples(conversations):
    """
    每一則 assistant 回覆都展開成一筆 training example：

        prompt     = system + 這則回覆之前的對話（最多 MAX_CONTEXT_MESSAGES 則）
        completion = 這則 assistant 回覆

    例如一段 3 輪對話 U1 A1 U2 A2 U3 A3 會展開成 3 筆：

        [U1]             → A1
        [U1 A1 U2]       → A2
        [U1 A1 U2 A2 U3] → A3
    """

    examples = []
    system_message = {"role": "system", "content": SYSTEM_PROMPT}

    for messages in conversations:
        for i, message in enumerate(messages):
            if message["role"] != "assistant":
                continue

            history = messages[:i][-MAX_CONTEXT_MESSAGES:]

            # 截斷後的上下文必須從 user 開始。
            if history and history[0]["role"] != "user":
                history = history[1:]

            examples.append(
                {
                    "prompt": [system_message] + history,
                    "completion": [message],
                }
            )

    return examples


def save_jsonl(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# =============================================================================
# 3. 主流程
# =============================================================================

def main():

    conversations = load_raw_conversations()
    if not conversations:
        raise RuntimeError(f"{RAW_DIR} 中沒有可用的對話。")

    turns = sum(len(m) // 2 for m in conversations)
    print(f"合計：{len(conversations)} 段對話，{turns} 輪")

    # 輸出加上 system prompt 的完整對話，方便檢視語料。
    save_jsonl(
        [
            {"messages": [{"role": "system", "content": SYSTEM_PROMPT}] + m}
            for m in conversations
        ],
        CHAT_FILE,
    )

    # 以整段對話為單位切分。
    random.Random(SEED).shuffle(conversations)
    n_valid = max(1, round(len(conversations) * VALID_RATIO))
    valid_conversations = conversations[:n_valid]
    train_conversations = conversations[n_valid:]

    train_examples = build_examples(train_conversations)
    valid_examples = build_examples(valid_conversations)

    save_jsonl(train_examples, OUTPUT_DIR / "train.jsonl")
    save_jsonl(valid_examples, OUTPUT_DIR / "valid.jsonl")

    print(
        f"train：{len(train_conversations)} 段對話 → {len(train_examples)} 筆 examples"
    )
    print(
        f"valid：{len(valid_conversations)} 段對話 → {len(valid_examples)} 筆 examples"
    )
    print(f"輸出完成：{OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
