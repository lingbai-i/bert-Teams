"""评估模块：分类指标、混淆矩阵与 bad case 导出。

支持两种输入：
- .txt    每行 `文本\t标签ID`（train/dev/test 同格式）
- .jsonl  每行 {"query": 文本, "intent": 标签名}（人工评估集）

用法：
    python -m src.evaluate --ckpt checkpoints/intent_bert/best.pt --input data/test.txt
    python -m src.evaluate --ckpt checkpoints/intent_bert/best.pt \
        --input data/eval/test_queries.jsonl --output-dir outputs/eval_human
"""
from pathlib import Path
from typing import List, Tuple

import argparse
import json
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from transformers import AutoTokenizer

from src.config import CKPT_DIR, CLASS_FILE, MAX_LEN, MODEL_DIR, OUTPUT_DIR
from src.data import load_labels, load_txt
from src.model import load_classifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def load_any(path: Path, label_names: List[str]) -> Tuple[List[str], List[int]]:
    """按文件后缀自动识别数据格式并加载。

    参数:
        path: .txt 或 .jsonl 数据文件。
        label_names: 标签名列表（class.txt 顺序），jsonl 按名字映射为 ID。

    返回:
        (文本列表, 标签 ID 列表)。

    异常:
        ValueError: 不支持的文件后缀，或 jsonl 中出现未知标签名。
    """
    if path.suffix == ".txt":
        return load_txt(path)
    if path.suffix == ".jsonl":
        name2id = {name: i for i, name in enumerate(label_names)}
        texts, labels = [], []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj["intent"] not in name2id:
                    raise ValueError(f"未知标签名：{obj['intent']}，请核对 class.txt")
                texts.append(obj["query"])
                labels.append(name2id[obj["intent"]])
        return texts, labels
    raise ValueError(f"不支持的数据格式：{path.suffix}（仅支持 .txt / .jsonl）")


def predict_all(model, tokenizer, texts: List[str], device: str,
                batch_size: int = 64) -> List[int]:
    """对文本列表批量推理，返回预测标签 ID 列表。"""
    preds = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            enc = tokenizer(batch_texts, truncation=True, max_length=MAX_LEN,
                            padding=True, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            preds.extend(logits.argmax(-1).cpu().tolist())
    return preds


def save_confusion_matrix(cm: np.ndarray, label_names: List[str], out_path: Path) -> None:
    """将混淆矩阵绘制为 PNG 图片（坐标轴使用英文标签名，避免中文字体缺失）。"""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(label_names)), label_names, rotation=45, ha="right")
    ax.set_yticks(range(len(label_names)), label_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(args) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    label_names = load_labels(CLASS_FILE)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = load_classifier(MODEL_DIR, Path(args.ckpt), len(label_names), device)

    texts, gold = load_any(Path(args.input), label_names)
    preds = predict_all(model, tokenizer, texts, device, args.bs)

    acc = accuracy_score(gold, preds)
    macro_f1 = f1_score(gold, preds, average="macro")
    report = classification_report(gold, preds, target_names=label_names, digits=4)
    cm = confusion_matrix(gold, preds)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.txt").write_text(
        f"accuracy = {acc:.4f}\nmacro_f1 = {macro_f1:.4f}\n\n{report}",
        encoding="utf-8")
    save_confusion_matrix(cm, label_names, out_dir / "confusion_matrix.png")
    with open(out_dir / "badcases.txt", "w", encoding="utf-8") as f:
        for text, g, p in zip(texts, gold, preds):
            if g != p:
                f.write(f"{label_names[g]}\t{label_names[p]}\t{text}\n")

    logger.info(f"样本数={len(texts)} accuracy={acc:.4f} macro_f1={macro_f1:.4f}")
    logger.info(f"bad case 数={int((np.array(gold) != np.array(preds)).sum())}，"
                f"结果已写入 {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="意图分类模型评估")
    ap.add_argument("--ckpt", default=str(CKPT_DIR / "intent_bert" / "best.pt"))
    ap.add_argument("--input", required=True, help=".txt 或 .jsonl 数据文件")
    ap.add_argument("--output-dir", default=str(OUTPUT_DIR / "eval"))
    ap.add_argument("--bs", type=int, default=64)
    main(ap.parse_args())
