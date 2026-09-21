"""分析企业客服意图数据的分布、重复和模板同源风险。

从 config 读取默认路径，保留原始文本进行精确重复统计；边界短语归一化
仅用于启发式模板分析，不修改数据或训练逻辑。命令行输出 JSON 证据及
matplotlib 图表到 outputs/eda，供同目录人工撰写的报告引用。
"""
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
import hashlib
import json
import platform

import matplotlib
import numpy as np

from . import config

# 从样本观察到的边界短语；长串优先，避免先剥离“请问”后残留“一下”。
# 这些是分析规则而非生成器真值，不删除正文中间内容，也不合并同义词。
PREFIXES = tuple(sorted([
    "打扰了", "咨询个事", "想了解下", "请问一下", "我想问下", "麻烦问下",
    "请教一下", "想知道", "想问下", "咨询下", "问个事", "关于", "你好",
    "请问", "那个", "在吗", "哎", "咱们", "咱",
], key=lambda value: (-len(value), value)))
SUFFIXES = tuple(sorted([
    "需要什么材料", "怎么办理", "怎么申请", "如何办理", "怎么操作", "麻烦说下",
    "是什么", "怎么办", "怎么弄", "要多久", "的政策", "的规则", "谢谢啦",
    "麻烦了", "谢谢", "辛苦", "流程", "要求", "条件", "规定", "时间",
    "审批", "在哪", "哈", "？", "?", "~", "！", "!", "。",
], key=lambda value: (-len(value), value)))


def read_rows(path: Path, class_count: int) -> List[Tuple[str, int]]:
    """读取 UTF-8 文本与整数标签，返回原始文本/标签元组列表。

    path 为输入文件，class_count 为合法标签上界（不含）。保留文本空格；
    空行、空文本、缺失分隔符、非法标签或空文件均抛出含位置的 ValueError。
    文件访问错误原样抛出，不静默跳行，以免改变统计分母。
    """
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            text, label = line.rsplit("\t", 1)
            label = int(label)
            if not text.strip() or not 0 <= label < class_count:
                raise ValueError("空文本或标签超界")
        except ValueError as exc:
            raise ValueError(f"{path.name}:{line_no}: 无效的文本/标签记录") from exc
        rows.append((text, label))
    if not rows:
        raise ValueError(f"{path.name}: 空数据集")
    return rows


def core_text(text: str) -> str:
    """逐次移除非空字符串 text 的已知前后缀，返回启发式核心短语。

    只匹配字符串边界；剥离将导致空串时停止。规则可能过度归并语义，
    返回值不能用于判定标签真值或充当实际生成模板 ID；无文件副作用。
    """
    core = text
    while True:
        previous = core
        for prefix in PREFIXES:
            if core.startswith(prefix) and len(core) > len(prefix):
                core = core[len(prefix):]
                break
        for suffix in SUFFIXES:
            if core.endswith(suffix) and len(core) > len(suffix):
                core = core[:-len(suffix)]
                break
        if core == previous:
            return core


def summarize(rows: List[Tuple[str, int]], class_count: int) -> Dict:
    """统计非空 rows 的类别、字符长度和重复，class_count 决定零计数类别。

    返回 JSON 可序列化字典。长度用 Python len（含空格/标点），分位数
    使用 numpy 线性插值；重复多余行=N-唯一文本数。空输入抛 ValueError。
    """
    if not rows:
        raise ValueError("无法统计空数据集")
    counts = Counter(label for _, label in rows)
    texts = Counter(text for text, _ in rows)
    labels = defaultdict(set)
    for text, label in rows:
        labels[text].add(label)
    lengths = np.array([len(text) for text, _ in rows])
    quantiles = dict(zip(["min", "p25", "median", "p75", "p95", "p99", "max"],
                        np.percentile(lengths, [0, 25, 50, 75, 95, 99, 100]).tolist()))
    return {
        "rows": len(rows), "class_counts": [counts[i] for i in range(class_count)],
        "unique_texts": len(texts), "duplicate_excess_rows": len(rows) - len(texts),
        "duplicate_excess_rate": (len(rows) - len(texts)) / len(rows),
        "conflicting_texts": sum(len(value) > 1 for value in labels.values()),
        "length": {**quantiles, "mean": float(lengths.mean()),
                   "std_population": float(lengths.std())},
    }


def overlap_stats(source: List[Tuple[str, int]], target: List[Tuple[str, int]]) -> Dict:
    """计算目标行在源文本集合中的命中率，返回计数和比例字典。

    source/target 为文本和标签列表；同一目标文本出现多次则逐行计数。
    标签不属于源文本的任何标签时记为分歧；空 target 抛 ValueError。
    """
    if not target:
        raise ValueError("目标集不能为空")
    labels = defaultdict(set)
    for text, label in source:
        labels[text].add(label)
    matched = [(text, label) for text, label in target if text in labels]
    return {
        "target_rows": len(target), "matched_rows": len(matched),
        "matched_unique_texts": len({text for text, _ in matched}),
        "matched_rate": len(matched) / len(target),
        "label_disagreement_rows": sum(label not in labels[text] for text, label in matched),
    }


def analyze(data_dir: Path) -> Tuple[Dict, Dict]:
    """分析 data_dir 的三个划分及人工 JSONL，返回证据字典和原始行字典。

    标签表须含九个唯一标签。记录输入 SHA-256、跨集重复、归一化覆盖及
    冲突实例；人工集的未知标签保留原值报告，不擅自映射。输入错误抛出。
    """
    classes = (data_dir / "class.txt").read_text(encoding="utf-8").splitlines()
    if len(classes) != 9 or len(set(classes)) != 9:
        raise ValueError("class.txt 必须包含九个唯一标签")
    splits = {name: read_rows(data_dir / f"{name}.txt", len(classes))
              for name in ("train", "dev", "test")}
    cores = {name: [(core_text(text), label) for text, label in rows]
             for name, rows in splits.items()}
    pairs = (("train", "dev"), ("train", "test"), ("dev", "test"))
    human = [json.loads(line) for line in (data_dir / "eval/test_queries.jsonl")
             .read_text(encoding="utf-8").splitlines()]
    human_counts = Counter(item["intent"] for item in human)
    train_texts = {text for text, _ in splits["train"]}
    files = ["class.txt", "train.txt", "dev.txt", "test.txt", "eval/test_queries.jsonl"]
    evidence = {
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "matplotlib": matplotlib.__version__},
        "input_sha256": {name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest()
                         for name in files},
        "classes": classes,
        "splits": {name: summarize(rows, len(classes)) for name, rows in splits.items()},
        "exact_overlap": {f"{a}_to_{b}": overlap_stats(splits[a], splits[b]) for a, b in pairs},
        "core_overlap": {f"{a}_to_{b}": overlap_stats(cores[a], cores[b]) for a, b in pairs},
        "core_unique_counts": {name: len({text for text, _ in rows}) for name, rows in cores.items()},
        "normalization_rules": {"prefixes": PREFIXES, "suffixes": SUFFIXES},
        "human": {"rows": len(human), "class_counts": dict(human_counts),
                  "missing_classes": [label for label in classes if label not in human_counts],
                  "unknown_labels": sorted(set(human_counts) - set(classes)),
                  "exact_train_matches": sum(item["query"] in train_texts for item in human)},
    }
    combined = defaultdict(lambda: {"labels": set(), "splits": Counter()})
    groups = defaultdict(lambda: {"texts": set(), "splits": Counter(), "labels": Counter()})
    for split, rows in splits.items():
        for text, label in rows:
            combined[text]["labels"].add(label)
            combined[text]["splits"][split] += 1
            group = groups[core_text(text)]
            group["texts"].add(text)
            group["splits"][split] += 1
            group["labels"][classes[label]] += 1
    conflicts = [{"text": text, "labels": sorted(value["labels"]),
                  "splits": dict(value["splits"])} for text, value in sorted(combined.items())
                 if len(value["labels"]) > 1]
    evidence["conflicting_text_count_all_splits"] = len(conflicts)
    evidence["conflicting_text_examples"] = conflicts[:20]
    evidence["top_core_groups"] = [
        {"core": core, "unique_variants": len(group["texts"]),
         "split_rows": dict(group["splits"]), "labels": dict(group["labels"]),
         "examples": sorted(group["texts"])[:6]}
        for core, group in sorted(groups.items(), key=lambda item: (-len(item[1]["texts"]), item[0]))[:20]
    ]
    return evidence, splits


def make_figures(evidence: Dict, splits: Dict, output: Path) -> None:
    """根据证据与原始行绘制四张中文 PNG 至 output，返回 None。

    要求系统存在中文字体；缺少字体时抛 RuntimeError，避免生成缺字图。
    使用 Agg 离线绘图，每次写入同名图表并关闭 figure，不弹出窗口。
    """
    matplotlib.use("Agg")
    from matplotlib import font_manager, pyplot as plt

    available = {font.name for font in font_manager.fontManager.ttflist}
    fonts = [name for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "WenQuanYi Zen Hei")
             if name in available]
    if not fonts:
        raise RuntimeError("绘图需要微软雅黑、黑体或 Noto Sans CJK SC 中文字体")
    plt.rcParams.update({"font.sans-serif": fonts, "axes.unicode_minus": False,
                         "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    names = ["考勤", "IT", "财务", "人事", "行政", "工程", "法务", "销售", "其他/闲聊"]
    colors = ["#2563A6", "#E09836", "#399787"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout="constrained")
    for ax, (split, rows), color in zip(axes, splits.items(), colors):
        counts = evidence["splits"][split]["class_counts"]
        bars = ax.barh(names, np.array(counts) / len(rows) * 100, color=color)
        ax.bar_label(bars, labels=[f"{n:,}" for n in counts], padding=3, fontsize=9)
        ax.set(xlim=(0, 16), xlabel="占本划分比例（%）；柱旁标注条数", title=f"{split} · {len(rows):,} 条")
        ax.invert_yaxis()
    fig.suptitle("类别分布：训练集均衡，验证集缺少其他/闲聊类", fontsize=17)
    fig.savefig(output / "class_distribution.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5.5), layout="constrained")
    max_len = max(len(text) for rows in splits.values() for text, _ in rows)
    for (split, rows), color in zip(splits.items(), colors):
        lengths = [len(text) for text, _ in rows]
        ax.hist(lengths, bins=np.arange(0.5, max_len + 1.5), weights=np.full(len(rows), 100 / len(rows)),
                histtype="step", linewidth=2.2, color=color, label=f"{split} (n={len(rows):,})")
    ax.set(xlabel="字符数（含标点和空格，不等同于 token 数）", ylabel="本划分样本占比（%）",
           title="文本长度直方图：三个划分均以短问句为主", xticks=range(1, max_len + 1, 2))
    ax.legend()
    fig.savefig(output / "length_histogram.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for ax, metric, title in zip(axes, ("exact_overlap", "core_overlap"),
                                 ("原始文本完全相同", "启发式核心短语相同（不等同于泄漏）")):
        values = [100 * evidence[metric][f"train_to_{split}"]["matched_rate"] for split in ("dev", "test")]
        bars = ax.bar(["验证集 → 训练集", "测试集 → 训练集"], values, color=colors[1:])
        ax.bar_label(bars, fmt="%.2f%%", padding=4)
        ax.set(ylim=(0, max(values) * 1.2), ylabel="目标划分中命中的样本行占比（%）", title=title)
    fig.suptitle("区分直接重复与模板/主题同源", fontsize=16)
    fig.savefig(output / "overlap_comparison.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
    counts = [evidence["human"]["class_counts"].get(label, 0) for label in evidence["classes"]]
    unknown = sum(n for label, n in evidence["human"]["class_counts"].items() if label not in evidence["classes"])
    bars = ax.bar(names + ["体系外标签"], counts + [unknown], color=[colors[0]] * 9 + ["#B74E4E"])
    ax.bar_label(bars)
    ax.set(ylim=(0, max(counts + [unknown]) + 1), ylabel="样本条数", title="人工评估集：类别覆盖与标签口径仍需补齐")
    fig.savefig(output / "human_coverage.png", dpi=160)
    plt.close(fig)


def main() -> None:
    """读取可选数据/输出目录参数并写证据及图表，异常直接向调用端传播。"""
    parser = argparse.ArgumentParser(description="企业客服数据 EDA（不依赖模型）")
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=config.OUTPUT_DIR / "eda")
    args = parser.parse_args()
    evidence, splits = analyze(args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "statistics.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    make_figures(evidence, splits, args.output_dir)
    print("EDA 统计与四张图表已生成；统计快照：statistics.json")


if __name__ == "__main__":
    main()
