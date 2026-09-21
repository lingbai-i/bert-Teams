"""合成集与人工集评估结果对比工具。

读取两次 `python -m src.evaluate` 产出的 metrics.json，输出答辩用的对比表：
总体准确率与 macro-F1 的差值、逐类 F1 的升降、两侧最主要的混淆对。
本脚本只读指标文件，不加载模型与数据，因此可以在任意机器上复核他人产出的
outputs/eval_test 与 outputs/eval_human，避免结论与数字脱节。

用法：
    python scripts/compare_eval.py --synthetic outputs/eval_test --human outputs/eval_human
    python scripts/compare_eval.py --synthetic outputs/eval_test --human outputs/eval_human \
        --out docs/eval_comparison.md
"""
from pathlib import Path
from typing import Dict, List

import argparse
import json

# 逐类 F1 的升降排序阈值：低于该值视为基本持平，避免把训练噪声当成结论
F1_FLAT_TOLERANCE = 0.005


def load_metrics(directory: Path) -> Dict[str, object]:
    """读取某个评估输出目录下的 metrics.json。

    参数:
        directory: evaluate 的 --output-dir 目录。

    返回:
        metrics.json 解析后的字典，含 accuracy / macro_f1 / per_class_f1 / confusion_pairs 等。

    异常:
        FileNotFoundError: 目录下不存在 metrics.json，说明该目录不是 evaluate 的产物
            （旧版本 evaluate 不写 metrics.json，需重新运行一次评估）。
    """
    path = directory / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} 不存在：请先运行 python -m src.evaluate --output-dir {directory}")
    return json.loads(path.read_text(encoding="utf-8"))


def _delta_text(synthetic_value: float, human_value: float) -> str:
    """把两个人集上的同一指标折算成带方向的差值文本（百分点）。

    参数:
        synthetic_value: 合成集指标值，0~1。
        human_value: 人工集指标值，0~1。

    返回:
        形如 "-9.42pp" 的差值文本，正负号表示人工集相对合成集的升降。
    """
    return f"{(human_value - synthetic_value) * 100:+.2f}pp"


def format_report(synthetic: Dict[str, object], human: Dict[str, object]) -> str:
    """生成 Markdown 格式的对比报告。

    参数:
        synthetic: 合成测试集的 metrics 字典。
        human: 人工评估集的 metrics 字典。

    返回:
        可直接写入文件的 Markdown 文本，含总体指标表、逐类 F1 表与主要混淆对。
    """
    lines: List[str] = ["# 合成集 vs 人工集 评估对比", ""]
    lines.append("> 由 `python scripts/compare_eval.py` 依据两侧 metrics.json 自动生成，请勿手工修改数字。")
    lines.append("")

    lines.append("## 一、总体指标")
    lines.append("")
    lines.append("| 指标 | 合成集 | 人工集 | 差值（人工 - 合成） |")
    lines.append("|------|--------|--------|---------------------|")
    for key, name in (("n", "样本数"), ("accuracy", "accuracy"), ("macro_f1", "macro-F1"),
                      ("weighted_f1", "weighted-F1"), ("badcase", "bad case 数"),
                      ("badcase_rate", "错误率")):
        left, right = synthetic[key], human[key]
        if key in {"accuracy", "macro_f1", "weighted_f1", "badcase_rate"}:
            lines.append(f"| {name} | {left:.4f} | {right:.4f} | {_delta_text(left, right)} |")
        else:
            lines.append(f"| {name} | {left} | {right} | {right - left:+d} |")
    lines.append("")

    lines.append("## 二、逐类 F1（按人工集 F1 升序，人工集上表现最弱的类别排在最前）")
    lines.append("")
    lines.append("| 类别 | 合成集 F1 | 人工集 F1 | 差值 | 判定 |")
    lines.append("|------|-----------|-----------|------|------|")
    per_class_syn: Dict[str, float] = synthetic["per_class_f1"]
    per_class_human: Dict[str, float] = human["per_class_f1"]
    for name in sorted(per_class_human, key=lambda k: per_class_human[k]):
        syn_f1, hum_f1 = per_class_syn[name], per_class_human[name]
        gap = hum_f1 - syn_f1
        verdict = "持平" if abs(gap) < F1_FLAT_TOLERANCE else ("退化" if gap < 0 else "提升")
        lines.append(f"| {name} | {syn_f1:.4f} | {hum_f1:.4f} | {gap * 100:+.2f}pp | {verdict} |")
    lines.append("")

    lines.append("## 三、主要混淆对（各取前 5 对）")
    lines.append("")
    for title, metrics in (("合成集", synthetic), ("人工集", human)):
        lines.append(f"### {title}（错误数 {metrics['badcase']}，错误率 {metrics['badcase_rate']:.2%}）")
        lines.append("")
        lines.append("| 混淆对（真实 -> 预测） | 条数 | 占错误比 |")
        lines.append("|------------------------|------|----------|")
        pairs = metrics["confusion_pairs"]
        if not pairs:
            lines.append("| 无错误 | 0 | - |")
        for pair in pairs[:5]:
            lines.append(f"| {pair['gold']} -> {pair['pred']} | {pair['count']} | {pair['share_of_errors']:.1%} |")
        lines.append("")
    return "\n".join(lines)


def main(args) -> None:
    synthetic = load_metrics(Path(args.synthetic))
    human = load_metrics(Path(args.human))
    report = format_report(synthetic, human)
    print(report)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report + "\n", encoding="utf-8", newline="\n")
        print(f"已写入 {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="合成集与人工集评估结果对比")
    ap.add_argument("--synthetic", default="outputs/eval_test", help="合成测试集的评估输出目录")
    ap.add_argument("--human", default="outputs/eval_human", help="人工评估集的评估输出目录")
    ap.add_argument("--out", default="", help="可选的 Markdown 输出路径")
    main(ap.parse_args())
