"""
plot_loss.py

讀取 outputs/love_talk_lora/trainer_state.json 的 log_history，
畫出 Training / Validation Loss，輸出 images/loss_curve.png。
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


STATE_FILE = Path("outputs/love_talk_lora/trainer_state.json")
PLOT_FILE = Path("images/loss_curve.png")


def main():

    if not STATE_FILE.is_file():
        raise FileNotFoundError(f"找不到：{STATE_FILE.resolve()}\n請先執行 train_qlora.py。")

    with STATE_FILE.open(encoding="utf-8") as f:
        state = json.load(f)

    logs = state.get("log_history", [])
    train_loss = [(x["step"], x["loss"]) for x in logs if "loss" in x]
    valid_loss = [(x["step"], x["eval_loss"]) for x in logs if "eval_loss" in x]

    PLOT_FILE.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    if train_loss:
        plt.plot(*zip(*train_loss), label="Training Loss", linewidth=1.8)
    if valid_loss:
        plt.plot(*zip(*valid_loss), marker="o", label="Validation Loss", linewidth=1.8)

    best_step = state.get("best_global_step")
    if best_step:
        plt.axvline(best_step, color="gray", linestyle="--", alpha=0.6, label=f"Best (step {best_step})")

    plt.xlabel("Optimizer Step")
    plt.ylabel("Loss")
    plt.title("QLoRA Training and Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Loss 圖表：{PLOT_FILE.resolve()}")


if __name__ == "__main__":
    main()
