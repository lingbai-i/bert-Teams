"""评估模块：分类指标、混淆矩阵与 bad case 归类分析。

支持两种输入：
- .txt    每行 `文本\t标签ID`（train/dev/test 同格式，模板合成分布）
- .jsonl  每行 {"query": 文本, "intent": 标签名}（人工评估集）

每次评估在 --output-dir 下产出四件套：
- report.txt             accuracy / macro-F1 / 逐类 precision-recall-f1 报告
- confusion_matrix.png   混淆矩阵热力图（原始计数，坐标轴用英文标签名以避开中文字体缺失）
- badcases.txt           错误样本按「真实标签 -> 预测标签」归类，含高频词证据与人工归因
- metrics.json           机器可读指标，供 scripts/compare_eval.py 对比合成集与人工集

依赖关系：
- 标签体系与数据读取复用 src.data，模型构建复用 src.model，路径统一取自 src.config，
  本模块不另行定义路径常量，保证评估与其他环节读的是同一份数据与权重。
- 人工归因文本放在 data/eval/badcase_notes.md，由本模块追加到 badcases.txt 第三节。

用法：
    python -m src.evaluate --input data/test.txt --output-dir outputs/eval_test
    python -m src.evaluate --input data/eval/test_queries.jsonl --output-dir outputs/eval_human
"""
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import argparse
import collections
import datetime
import hashlib
import json
import logging
import re

import jieba
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from transformers import AutoTokenizer

from src.config import (
    BADCASE_NOTES_FILE, BEST_CKPT, CLASS_FILE, MAX_LEN, MODEL_DIR, OUTPUT_DIR, STOPWORD_FILE,
)
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
        ValueError: 不支持的文件后缀，或 jsonl 中出现 class.txt 里没有的标签名。
            标签名拼错属于数据问题，此处直接失败以便定位，不做静默丢弃或改名处理。
    """
    if path.suffix == ".txt":
        return load_txt(path)
    if path.suffix == ".jsonl":
        name2id = {name: i for i, name in enumerate(label_names)}
        texts, labels = [], []
        with open(path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj["intent"] not in name2id:
                    raise ValueError(
                        f"{path} 第 {lineno} 行标签名未知：{obj['intent']}，"
                        f"合法标签见 {CLASS_FILE}")
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


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """计算权重文件的 SHA256，用于在产物里钉住本次评估的模型版本。

    背景：models/best.pt 会被反复替换（换权重后指标变化，但文件名不变），
    只记录路径无法回溯某个指标到底出自哪份权重。项目实践中曾出现评估中途
    权重被覆盖、指标前后不一致的情况，因此把哈希写进产物作为版本凭证。

    参数:
        path: 待计算的文件路径。
        chunk_size: 分块读取字节数，避免 390MB 权重一次性读入内存。

    返回:
        十六进制 SHA256 字符串。
    """
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def load_stopwords(path: Path) -> set:
    """读取停用词表，用于过滤高频词统计中的噪声词。

    参数:
        path: 停用词文件路径（每行一个词，沿用 TF-IDF 路线同一份 data/stopwords.txt）。

    返回:
        停用词集合；文件缺失时返回空集合（高频词统计仍可完成，只是噪声更多）。
    """
    if not path.exists():
        logger.warning(f"未找到停用词表 {path}，高频词统计将不过滤停用词")
        return set()
    return {w.strip() for w in path.read_text(encoding="utf-8").splitlines() if w.strip()}


def top_terms(texts: List[str], stopwords: set, topk: int = 8) -> List[Tuple[str, int]]:
    """统计一组误判文本的高频词，作为“模型抓到了什么线索”的证据。

    统计前过滤停用词、单字词与非文字符号：单字在中文里虽然携带语义，
    但在类别互串的场景下绝大多数是“的/了/吗”一类虚词，保留只会淹没有效信息。

    参数:
        texts: 待统计的误判文本。
        stopwords: 停用词集合。
        topk: 最多返回的词条数。

    返回:
        [(词, 出现次数), ...]，按出现次数降序。
    """
    counter: collections.Counter = collections.Counter()
    for text in texts:
        for word in jieba.lcut(text):
            word = word.strip()
            if len(word) < 2 or word in stopwords:
                continue
            if not re.search(r"[\u4e00-\u9fa5A-Za-z0-9]", word):
                continue
            counter[word] += 1
    return counter.most_common(topk)


def write_badcases(texts: List[str], gold: Sequence[int], preds: Sequence[int],
                   label_names: List[str], dataset: Path, ckpt: Path, out_path: Path,
                   ckpt_sha: str, max_examples: int = 5) -> Dict[str, object]:
    """按「真实标签 -> 预测标签」归类错误样本并写出 badcases.txt。

    输出分三节，逐层从“现象”走到“原因”：
    1. 混淆对总览：错误数降序，给出每一对占全部错误的比重，用于判断主要矛盾；
    2. 混淆对明细：逐对给出高频词证据与原句示例，支撑后续归因判断；
    3. 人工归因分析：追加 data/eval/badcase_notes.md 的正文（若存在）。
    用“混淆对”而非“逐条罗列”组织，是因为答辩要回答的是“哪些类别容易混淆”，
    逐条罗列只能看到零散个案，无法体现错误的结构性分布。

    参数:
        texts: 与 gold/preds 等长的原始文本。
        gold: 真实标签 ID 序列。
        preds: 预测标签 ID 序列。
        label_names: 标签名列表，下标即标签 ID。
        dataset: 评估数据文件路径，写入报告头便于回溯。
        ckpt: 本次评估所用权重路径，写入报告头便于回溯。
        out_path: badcases.txt 输出路径。
        ckpt_sha: 权重文件的 SHA256，写入报告头以便回溯模型版本。
        max_examples: 每个混淆对最多列出的示例条数。

    返回:
        摘要字典（样本数、错误数、错误率、混淆对列表），供 metrics.json 复用，
        避免指标与报告两处各算一遍导致口径不一致。
    """
    groups: Dict[Tuple[int, int], List[int]] = collections.defaultdict(list)
    for idx, (g, p) in enumerate(zip(gold, preds)):
        if g != p:
            groups[(g, p)].append(idx)
    ordered = sorted(groups.items(),
                     key=lambda kv: (-len(kv[1]), label_names[kv[0][0]], label_names[kv[0][1]]))

    total = len(texts)
    bad = sum(len(idxs) for _, idxs in ordered)
    stopwords = load_stopwords(STOPWORD_FILE)

    lines: List[str] = ["# bad case 归类分析", ""]
    lines.append(f"生成时间: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    lines.append(f"评估数据: {dataset}")
    lines.append(f"权重文件: {ckpt}（sha256 {ckpt_sha[:16]}）")
    lines.append(f"样本数: {total}    错误数: {bad}    错误率: {bad / max(total, 1) * 100:.2f}%")
    lines.append("")

    if bad == 0:
        lines.append("本次评估全部样本预测正确，无 bad case。")
    else:
        lines.append("## 一、混淆对总览（按错误数降序）")
        lines.append("")
        lines.append("错误数\t占错误比\t混淆对")
        for (g, p), idxs in ordered:
            lines.append(f"{len(idxs)}\t{len(idxs) / bad * 100:.1f}%\t{label_names[g]} -> {label_names[p]}")
        lines.append("")
        lines.append(f"## 二、混淆对明细（每对最多 {max_examples} 条示例）")
        lines.append("")
        for rank, ((g, p), idxs) in enumerate(ordered, start=1):
            pair_texts = [texts[i] for i in idxs]
            terms = top_terms(pair_texts, stopwords)
            lines.append(f"### {rank}. {label_names[g]} -> {label_names[p]}    共 {len(idxs)} 条"
                         f"（占错误 {len(idxs) / bad * 100:.1f}%）")
            if terms:
                lines.append("高频词: " + "、".join(f"{w}({c})" for w, c in terms))
            for text in pair_texts[:max_examples]:
                lines.append(f"- {text}")
            lines.append("")

    if BADCASE_NOTES_FILE.exists():
        lines.append("## 三、人工归因分析")
        lines.append("")
        lines.append(f"> 来源：{BADCASE_NOTES_FILE}（人工撰写，随评估结果一起评审）")
        lines.append("")
        lines.append(BADCASE_NOTES_FILE.read_text(encoding="utf-8").strip())
        lines.append("")
    else:
        lines.append("## 三、人工归因分析")
        lines.append("")
        lines.append(f"未提供 {BADCASE_NOTES_FILE}，本节留空。")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return {
        "badcase": bad,
        "badcase_rate": round(bad / max(total, 1), 4),
        "confusion_pairs": [
            {
                "gold": label_names[g],
                "pred": label_names[p],
                "count": len(idxs),
                "share_of_errors": round(len(idxs) / max(bad, 1), 4),
            }
            for (g, p), idxs in ordered
        ],
    }


def main(args) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"设备: {device}")
    label_names = load_labels(CLASS_FILE)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = load_classifier(MODEL_DIR, Path(args.ckpt), len(label_names), device)

    dataset = Path(args.input)
    texts, gold = load_any(dataset, label_names)
    preds = predict_all(model, tokenizer, texts, device, args.bs)

    acc = accuracy_score(gold, preds)
    macro_f1 = f1_score(gold, preds, average="macro")
    weighted_f1 = f1_score(gold, preds, average="weighted")
    # 评估集可能不包含全部类别（如小规模人工评估集），
    # 必须显式传入 labels 让报告/混淆矩阵/per_class_f1 对齐完整标签体系
    all_ids = list(range(len(label_names)))
    report = classification_report(gold, preds, labels=all_ids,
                                   target_names=label_names, digits=4,
                                   zero_division=0)
    report_dict = classification_report(gold, preds, labels=all_ids,
                                        target_names=label_names, digits=4,
                                        zero_division=0, output_dict=True)
    cm = confusion_matrix(gold, preds, labels=all_ids)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.txt").write_text(
        f"accuracy = {acc:.4f}\nmacro_f1 = {macro_f1:.4f}\nweighted_f1 = {weighted_f1:.4f}\n\n{report}",
        encoding="utf-8", newline="\n")
    save_confusion_matrix(cm, label_names, out_dir / "confusion_matrix.png")
    ckpt_sha = file_sha256(Path(args.ckpt))
    logger.info(f"权重 sha256={ckpt_sha[:16]}（{args.ckpt}）")
    summary = write_badcases(texts, gold, preds, label_names, dataset, Path(args.ckpt),
                             out_dir / "badcases.txt", ckpt_sha, args.max_examples)

    metrics = {
        "input": str(dataset),
        "ckpt": str(args.ckpt),
        "ckpt_sha256": ckpt_sha,
        "device": device,
        "n": len(texts),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "badcase": summary["badcase"],
        "badcase_rate": summary["badcase_rate"],
        "confusion_pairs": summary["confusion_pairs"],
        "per_class_f1": {name: round(report_dict[name]["f1-score"], 4) for name in label_names},
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")

    logger.info(f"样本数={len(texts)} accuracy={acc:.4f} macro_f1={macro_f1:.4f}")
    logger.info(f"bad case 数={summary['badcase']}，结果已写入 {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="意图分类模型评估")
    ap.add_argument("--ckpt", default=str(BEST_CKPT),
                    help="模型权重路径，默认取 models/best.pt（权威权重）")
    ap.add_argument("--input", required=True, help=".txt 或 .jsonl 数据文件")
    ap.add_argument("--output-dir", default=str(OUTPUT_DIR / "eval"))
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--max-examples", type=int, default=5,
                    help="badcases.txt 中每个混淆对最多列出的示例条数")
    main(ap.parse_args())
