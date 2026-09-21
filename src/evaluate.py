"""评估脚本：在 test.txt 上评测训练好的意图分类模型。

输出整体指标、各类别 precision/recall/F1、混淆矩阵，并做拒答阈值分析：
统计不同阈值下被"判定为闲聊"的比例与剩余样本的准确率；
同时对 eval/test_queries.jsonl 中的超出范围查询验证拒答行为。
结果以 outputs/report.md 及 PNG 图表形式落盘。

依赖关系：
- 依赖 src.config、src.dataset、src.predictor；
- 通过 `python -m src.evaluate` 运行。
"""
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_recall_fscore_support)

from src.config import Config
from src.dataset import load_class_labels, load_tsv
from src.predictor import IntentPredictor

# Windows 下使用微软雅黑渲染中文
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def plot_confusion_matrix(cf: np.ndarray, labels: List[str], save_path: Path) -> None:
    """绘制并保存混淆矩阵热力图。

    参数：
        cf: 混淆矩阵，形状 (num_labels, num_labels)。
        labels: 类别名称列表，顺序与矩阵行列一致。
        save_path: PNG 输出路径。
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cf, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cf[i, j]), ha="center", va="center",
                    color="white" if cf[i, j] > cf.max() / 2 else "black", fontsize=8)
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title("测试集混淆矩阵")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_threshold_curve(
    thresholds: List[float], acc_accepted: List[float], reject_rates: List[float],
    save_path: Path,
) -> None:
    """绘制拒答阈值与"接受样本准确率 / 拒答率"关系曲线。

    参数：
        thresholds: 扫描的阈值序列。
        acc_accepted: 每个阈值下接受样本（最高概率不低于阈值）的准确率。
        reject_rates: 每个阈值下被拒答样本（最高概率低于阈值）的占比。
        save_path: PNG 输出路径。
    """
    fig, ax1 = plt.subplots(figsize=(9, 6))
    ax1.plot(thresholds, acc_accepted, marker="o", label="接受样本准确率", color="#2f6fb2")
    ax1.set_xlabel("拒答阈值")
    ax1.set_ylabel("接受样本准确率", color="#2f6fb2")
    ax1.tick_params(axis="y", labelcolor="#2f6fb2")

    ax2 = ax1.twinx()
    ax2.plot(thresholds, reject_rates, marker="s", label="拒答率", color="#d9534f")
    ax2.set_ylabel("拒答率", color="#d9534f")
    ax2.tick_params(axis="y", labelcolor="#d9534f")
    ax2.set_ylim(0, 1)

    ax1.set_title("拒答阈值与准确率/拒答率关系")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def threshold_analysis(
    probs: np.ndarray, true_labels: np.ndarray,
) -> Tuple[List[float], List[float], List[float]]:
    """扫描拒答阈值，返回 (阈值序列, 接受样本准确率, 拒答率)。

    参数：
        probs: 模型输出的 softmax 概率矩阵。
        true_labels: 真实标签数组。

    返回：
        三个长度一致的列表，分别对应阈值、接受样本准确率、拒答率。
    """
    max_probs = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    thresholds = [round(t, 2) for t in np.arange(0.0, 0.96, 0.05)]
    acc_accepted: List[float] = []
    reject_rates: List[float] = []
    for t in thresholds:
        accepted = max_probs >= t
        n_accepted = int(accepted.sum())
        if n_accepted == 0:
            acc_accepted.append(float("nan"))
        else:
            acc_accepted.append(float(accuracy_score(true_labels[accepted], preds[accepted])))
        reject_rates.append(1.0 - n_accepted / len(max_probs))
    return thresholds, acc_accepted, reject_rates


def evaluate_queries(predictor: IntentPredictor, eval_file: Path) -> List[Dict[str, object]]:
    """在 eval/test_queries.jsonl 上逐条预测，验证范围内查询命中与越界查询拒答。

    参数：
        predictor: 已加载的推理器。
        eval_file: 评测查询 JSONL 路径。

    返回：
        每条查询的预测结果列表（含判定是否正确的 ok 字段）。
    """
    results: List[Dict[str, object]] = []
    for line in eval_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        out = predictor.predict(item["query"])
        expected = item["intent"]
        if expected == "out_of_scope":
            ok = out["rejected"]
        else:
            ok = (not out["rejected"]) and out["intent"] == expected
        results.append({
            "query": item["query"],
            "expected": expected,
            "predicted": out["intent"],
            "confidence": round(out["confidence"], 4),
            "rejected": out["rejected"],
            "ok": ok,
        })
    return results


def main() -> None:
    """执行评估主流程并生成报告。"""
    cfg = Config()
    cfg.report_dir.mkdir(parents=True, exist_ok=True)
    id2label, _ = load_class_labels(cfg.class_file)

    predictor = IntentPredictor()
    test_texts, test_labels = load_tsv(cfg.test_file)
    print(f"测试集 {len(test_texts)} 条，开始推理")

    probs = predictor.predict_probs(test_texts)
    preds = probs.argmax(axis=1)
    true_labels = np.array(test_labels)

    accuracy = float(accuracy_score(true_labels, preds))
    macro_f1 = float(f1_score(true_labels, preds, average="macro", zero_division=0))
    print(f"整体准确率 {accuracy:.4f}，宏平均 F1 {macro_f1:.4f}")

    report = classification_report(
        true_labels, preds, labels=list(range(len(id2label))),
        target_names=id2label, digits=4, zero_division=0,
    )
    cf = confusion_matrix(true_labels, preds, labels=list(range(len(id2label))))
    per_class = precision_recall_fscore_support(
        true_labels, preds, labels=list(range(len(id2label))), zero_division=0,
    )

    thresholds, acc_accepted, reject_rates = threshold_analysis(probs, true_labels)
    cfg_threshold = cfg.reject_threshold
    idx_cfg = min(range(len(thresholds)), key=lambda i: abs(thresholds[i] - cfg_threshold))

    # ---- 图表 ----
    cm_path = cfg.report_dir / "confusion_matrix.png"
    curve_path = cfg.report_dir / "threshold_curve.png"
    plot_confusion_matrix(cf, id2label, cm_path)
    plot_threshold_curve(thresholds, acc_accepted, reject_rates, curve_path)

    # ---- 越界查询评测 ----
    eval_results = evaluate_queries(predictor, cfg.eval_file)
    eval_pass = sum(1 for r in eval_results if r["ok"])
    eval_table = "".join(
        f"| {r['query']} | {r['expected']} | {r['predicted']} | {r['confidence']} "
        f"| {'拒答' if r['rejected'] else '回答'} | {'通过' if r['ok'] else '失败'} |\n"
        for r in eval_results
    )

    # ---- 组装报告 ----
    lines = [
        "# 意图识别模型评估报告\n",
        f"- 基座模型：{cfg.model_name}",
        f"- 拒答阈值：{cfg.reject_threshold}（最大类别概率低于阈值时判定为闲聊并拒答）",
        f"- 测试集：{len(test_texts)} 条（{len(id2label)} 个类别）\n",
        "## 一、整体指标\n",
        f"| 指标 | 数值 |",
        "| --- | --- |",
        f"| 准确率（Accuracy） | {accuracy:.4f} |",
        f"| 宏平均 F1（Macro F1） | {macro_f1:.4f} |\n",
        "## 二、各类别指标\n",
        "| 类别 | Precision | Recall | F1 | 样本数 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for i, label in enumerate(id2label):
        lines.append(
            f"| {label} | {per_class[0][i]:.4f} | {per_class[1][i]:.4f} "
            f"| {per_class[2][i]:.4f} | {int(per_class[3][i])} |",
        )
    lines += [
        "\n## 三、混淆矩阵\n",
        f"![混淆矩阵](confusion_matrix.png)\n",
        "## 四、拒答阈值分析\n",
        "接受样本指最高类别概率不低于阈值的样本；拒答率指被判定为闲聊而拒绝回答的样本占比。\n",
        "| 阈值 | 接受样本准确率 | 拒答率 |",
        "| --- | --- | --- |",
    ]
    for t, acc, rr in zip(thresholds, acc_accepted, reject_rates):
        lines.append(f"| {t:.2f} | {acc:.4f} | {rr:.4f} |")
    lines += [
        f"\n配置阈值 {cfg.reject_threshold:.2f} 对应的接受样本准确率为 "
        f"{acc_accepted[idx_cfg]:.4f}，拒答率为 {reject_rates[idx_cfg]:.4f}。\n",
        "![阈值曲线](threshold_curve.png)\n",
        "## 五、越界查询评测（eval/test_queries.jsonl）\n",
        f"共 {len(eval_results)} 条，其中 {eval_pass} 条判定正确。"
        "越界查询（out_of_scope）应被拒答，范围内查询应命中对应意图。\n",
        "| 查询 | 期望意图 | 预测意图 | 最高置信度 | 行为 | 判定 |",
        "| --- | --- | --- | --- | --- | --- |",
        eval_table,
    ]
    report_path = cfg.report_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"报告已生成：{report_path}")


if __name__ == "__main__":
    main()
