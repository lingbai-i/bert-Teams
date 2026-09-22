"""合成语料清洗脚本：去重 + 按规则重新分类。

为什么需要清洗：
    合成语料的生成映射表存在主题重叠——同一个主题字符串被两个类别各标了一半
    （例如“病假工资”在 policy_attendance 与 hr_onboarding 下各有 314/328 条，
    两个类别几乎 50:50）。模型在这批样本上的期望上限只有 50%，指标既虚高又不可解释；
    此外语料里还有大量“同文本、异标签”的重复行（train.txt 1266 行）。
    评估阶段测得合成集 178 条 bad case 中 75.8% 落在这些双标签主题上，因此先清洗数据再谈指标。

分类依据：
    data/synthetic_label_rules.json 的规则表。每条规则带 basis 字段说明裁定来源
    （knowledge_base = 按 data/knowledge_base 对应部门文档归属；convention_decision = 按团队口径决策，
    用于那些知识库与已训练权重冲突的主题），另带 knowledge_base_note 记录知识库原文对照，
    便于复核与回滚。本脚本不自行发明标签，只执行规则表。

匹配与去重口径：
    - 规则按最长短语优先：同时命中“劳动仲裁”（-> hr_onboarding）与更短的规则时以更长的为准，
      避免长短规则互相覆盖。
    - 未命中任何规则的样本保持原标签不动：只清洗已确认有问题的地方，避免清洗本身引入新噪声。
    - 去重以（文本, 标签）为单位；重分类后标签一致的重复文本合并为一条。
    - 输出顺序保持输入顺序，便于与原文件逐行比对。

用法：
    python scripts/clean_synthetic.py --input data/raw/test.txt --output data/test.txt
    python scripts/clean_synthetic.py --input data/raw/train.txt --output data/train.txt
    python scripts/clean_synthetic.py --input data/raw/dev.txt --output data/dev.txt
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import argparse
import collections
import json
import sys

# scripts/ 目录不在包内，直接运行时需要把仓库根目录加入模块搜索路径才能复用 src.config 的路径常量
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CLASS_FILE, ROOT


def load_rules(path: Path) -> List[Dict[str, str]]:
    """读取重分类规则表并按短语长度降序排列。

    参数:
        path: rules JSON 路径，结构见 data/synthetic_label_rules.json。

    返回:
        规则列表，按 phrase 长度降序，保证最长优先匹配。

    异常:
        ValueError: 规则使用了 class.txt 中不存在的标签名（说明规则表与标签体系脱节）。
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    labels = load_label_names()
    for rule in data["rules"]:
        if rule["label"] not in labels:
            raise ValueError(f"规则 {rule['phrase']} 的标签 {rule['label']} 不在 class.txt 中")
    return sorted(data["rules"], key=lambda r: -len(r["phrase"]))


def load_label_names() -> List[str]:
    """读取 class.txt 标签名列表（下标即标签 ID）。

    返回:
        标签名列表。
    """
    return [line.strip() for line in CLASS_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]


def match_rule(text: str, rules: List[Dict[str, str]]) -> Optional[Dict[str, str]]:
    """返回文本命中的规则（最长短语优先）。

    参数:
        text: 原始问句。
        rules: 已按长度降序排列的规则列表。

    返回:
        命中的规则字典；未命中返回 None。
    """
    for rule in rules:
        if rule["phrase"] in text:
            return rule
    return None


def load_rows(path: Path) -> List[Tuple[str, int]]:
    """读取 `文本\\t标签ID` 数据文件。

    参数:
        path: 数据文件路径，兼容 CRLF/LF。

    返回:
        (文本, 标签 ID) 列表，跳过不含制表符的行。

    异常:
        ValueError: 标签 ID 不是整数（说明文件被写坏）。
    """
    rows: List[Tuple[str, int]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if "\t" not in line:
            continue
        text, label = line.rsplit("\t", 1)
        try:
            rows.append((text.strip(), int(label)))
        except ValueError as exc:
            raise ValueError(f"{path} 第 {lineno} 行标签不是整数：{label!r}") from exc
    return rows


def clean(rows: List[Tuple[str, int]], rules: List[Dict[str, str]],
          label_names: List[str]) -> Tuple[List[Tuple[str, int]], Dict[str, object]]:
    """执行重分类与去重。

    参数:
        rows: 原始 (文本, 标签 ID) 列表。
        rules: 规则列表（已按长度降序）。
        label_names: 标签名列表，下标即标签 ID。

    返回:
        (清洗后的行列表, 统计字典)。统计含每条规则命中与改动条数、去重条数、
        同文本异标签条数、清洗前后类别分布。
    """
    name2id = {name: i for i, name in enumerate(label_names)}
    per_rule = collections.Counter()
    relabeled = 0
    kept: List[Tuple[str, int]] = []
    seen: Dict[Tuple[str, int], int] = {}
    dup_drop = 0
    text_labels: Dict[str, set] = collections.defaultdict(set)
    before = collections.Counter(label for _, label in rows)

    for text, label in rows:
        rule = match_rule(text, rules)
        new_label = name2id[rule["label"]] if rule is not None else label
        if rule is not None:
            per_rule[rule["phrase"]] += 1
            if new_label != label:
                relabeled += 1
        key = (text, new_label)
        if key in seen:
            dup_drop += 1
            continue
        seen[key] = 1
        kept.append((text, new_label))
        text_labels[text].add(new_label)

    residual = {t: sorted(s) for t, s in text_labels.items() if len(s) > 1}
    after = collections.Counter(label for _, label in kept)
    stats = {
        "rows_in": len(rows),
        "rows_out": len(kept),
        "relabeled": relabeled,
        "duplicates_dropped": dup_drop,
        "residual_conflicts": len(residual),
        "residual_examples": [f"{t} -> {[label_names[i] for i in s]}" for t, s in list(residual.items())[:5]],
        "per_rule": dict(per_rule),
        "before": {label_names[k]: v for k, v in sorted(before.items())},
        "after": {label_names[k]: v for k, v in sorted(after.items())},
    }
    return kept, stats


def format_report(stats: Dict[str, object], rules: List[Dict[str, str]], src: Path,
                  dst: Path) -> str:
    """生成清洗报告文本（写入 --report 指定的文件）。

    参数:
        stats: clean() 返回的统计字典。
        rules: 规则列表。
        src: 输入文件路径。
        dst: 输出文件路径。

    返回:
        报告文本。
    """
    lines = ["# 合成语料清洗报告", "", f"输入: {src}", f"输出: {dst}", ""]
    lines.append(f"输入行数: {stats['rows_in']}")
    lines.append(f"重分类改动行数: {stats['relabeled']}")
    lines.append(f"去重丢弃行数: {stats['duplicates_dropped']}")
    lines.append(f"输出行数: {stats['rows_out']}")
    lines.append(f"残留同文本异标签数: {stats['residual_conflicts']}"
                 "（清洗目标为 0）")
    if stats["residual_examples"]:
        lines.append("  残留示例: " + "；".join(stats["residual_examples"]))
    lines.append("")
    lines.append("## 各规则命中情况")
    lines.append("")
    lines.append("| 规则短语 | 裁定标签 | 类型 | 命中行数 | 裁定依据 | knowledge_base 对照 |")
    lines.append("|----------|----------|------|----------|----------|---------------------|")
    for rule in sorted(rules, key=lambda r: -len(r["phrase"])):
        hits = stats["per_rule"].get(rule["phrase"], 0)
        if hits == 0:
            continue
        lines.append(f"| {rule['phrase']} | {rule['label']} | {rule['kind']} | {hits} "
                     f"| {rule['basis']} | {rule['knowledge_base_note']} |")
    lines.append("")
    lines.append("## 类别分布（清洗前 -> 清洗后）")
    lines.append("")
    lines.append("| 类别 | 清洗前 | 清洗后 | 变化 |")
    lines.append("|------|--------|--------|------|")
    for name, before_n in stats["before"].items():
        after_n = stats["after"].get(name, 0)
        lines.append(f"| {name} | {before_n} | {after_n} | {after_n - before_n:+d} |")
    return "\n".join(lines) + "\n"


def main(args) -> None:
    rules = load_rules(Path(args.rules))
    label_names = load_label_names()
    src, dst = Path(args.input), Path(args.output)
    rows = load_rows(src)
    kept, stats = clean(rows, rules, label_names)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("".join(f"{t}\t{lab}\n" for t, lab in kept), encoding="utf-8", newline="\n")
    report = format_report(stats, rules, src, dst)
    print(report)
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8", newline="\n")
        print(f"报告已写入 {report_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="合成语料去重与重分类")
    ap.add_argument("--input", required=True, help="原始数据文件（文本\\t标签ID）")
    ap.add_argument("--output", required=True, help="清洗后输出路径")
    ap.add_argument("--rules", default=str(ROOT / "data" / "synthetic_label_rules.json"),
                    help="重分类规则表 JSON")
    ap.add_argument("--report", default="", help="可选：清洗报告输出路径")
    main(ap.parse_args())
