"""数据清洗脚本：剔除标注冲突样本，产出干净数据集。

清洗策略（依据 docs/data_quality.md 的分析结论）：
1. 训练集：同一文本被打上多个不同标签（生成脚本短语库跨类重复所致），
   该文本的全部样本直接剔除；同文同标的重复行去重保留一条。
   不做"按短语库归属统一标签"，避免把人工假设注入标签。
2. dev/test：剔除与清洗后训练集存在标签冲突的样本（此类样本必错一边，
   留在评估集中只会污染指标）；同文同标重复行去重。
3. 原始 data/train|dev|test.txt 不做任何修改，清洗结果写入 data/cleaned/。

用法：
    python scripts/clean_data.py
"""
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CLASS_FILE, DATA_DIR
from src.data import load_labels, load_txt


def build_label_map(texts: List[str], labels: List[int]) -> Tuple[Dict[str, int], int]:
    """构建文本到唯一标签的映射，并统计冲突文本数。

    参数:
        texts: 文本列表。
        labels: 对应的标签 ID 列表。

    返回:
        (无冲突的 文本→标签 映射, 被剔除的冲突文本种数)。
        同一文本标签一致的多行只保留一条；标签冲突的文本整体不进入映射。
    """
    text2labels = defaultdict(set)
    for t, l in zip(texts, labels):
        text2labels[t].add(l)
    clean_map = {}
    conflict = 0
    for t, ls in text2labels.items():
        if len(ls) == 1:
            clean_map[t] = next(iter(ls))
        else:
            conflict += 1
    return clean_map, conflict


def clean_split(texts: List[str], labels: List[int],
                ref_map: Dict[str, int]) -> Tuple[List[str], List[int], dict]:
    """按参考标签映射清洗一个数据集。

    剔除规则：文本在 ref_map 中存在但标签不一致（冲突）；
    去重规则：同文同标只保留首次出现的一行。

    参数:
        texts: 文本列表。
        labels: 标签 ID 列表。
        ref_map: 训练集清洗后的 文本→标签 映射（评估集对齐用）。

    返回:
        (清洗后的文本列表, 标签列表, 清洗统计字典)。
    """
    out_t, out_l = [], []
    seen = set()
    dropped_conflict = dropped_dup = 0
    for t, l in zip(texts, labels):
        if t in ref_map and ref_map[t] != l:
            dropped_conflict += 1
            continue
        if t in seen:
            dropped_dup += 1
            continue
        seen.add(t)
        out_t.append(t)
        out_l.append(l)
    return out_t, out_l, {"dropped_conflict": dropped_conflict,
                          "dropped_dup": dropped_dup}


def write_txt(path: Path, texts: List[str], labels: List[int]) -> None:
    """以 `文本\t标签ID` 格式写出数据集。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for t, l in zip(texts, labels):
            f.write(f"{t}\t{l}\n")


def main() -> None:
    label_names = load_labels(CLASS_FILE)
    stats = {}

    tr_t, tr_l = load_txt(DATA_DIR / "train.txt")
    train_map, train_conflict = build_label_map(tr_t, tr_l)
    # 用无冲突映射重建训练集（顺序打乱程度与原文件一致即可，训练时本就会 shuffle）
    clean_tr_t = list(train_map.keys())
    clean_tr_l = [train_map[t] for t in clean_tr_t]
    stats["train"] = {
        "before": len(tr_t), "after": len(clean_tr_t),
        "conflict_texts_removed": train_conflict,
        "rows_removed": len(tr_t) - len(clean_tr_t),
    }

    for name in ["dev", "test"]:
        texts, labels = load_txt(DATA_DIR / f"{name}.txt")
        ct, cl, st = clean_split(texts, labels, train_map)
        st["before"] = len(texts)
        st["after"] = len(ct)
        stats[name] = st
        write_txt(DATA_DIR / "cleaned" / f"{name}.txt", ct, cl)

    write_txt(DATA_DIR / "cleaned" / "train.txt", clean_tr_t, clean_tr_l)

    # 清洗后校验：训练集内不应再有同文异标
    _, residual = build_label_map(clean_tr_t, clean_tr_l)
    stats["residual_conflict_texts"] = residual

    out = Path("outputs/eda/cleaning_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    for name in ["train", "dev", "test"]:
        texts, labels = load_txt(DATA_DIR / "cleaned" / f"{name}.txt")
        from collections import Counter
        dist = Counter(labels)
        print(f"{name}: {len(texts)} 条，各类别:",
              {label_names[k]: v for k, v in sorted(dist.items())})


if __name__ == "__main__":
    main()
