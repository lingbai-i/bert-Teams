"""数据加载与预处理模块。

负责读取意图分类数据集（class.txt / train.txt / dev.txt / test.txt），
将文本与标签编号转换为 BERT 输入特征，供训练与评估使用。

依赖关系：
- 依赖 src.config；
- 被 src/train.py、src/evaluate.py、src/infer.py 引用。
"""
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


def load_class_labels(class_file: Path) -> Tuple[List[str], dict]:
    """读取类别列表，返回 (id2label, label2id)。

    class.txt 每行一个类别名，行号（从 0 开始）即标签编号。
    """
    lines = class_file.read_text(encoding="utf-8").splitlines()
    labels = [line.strip() for line in lines if line.strip()]
    label2id = {name: idx for idx, name in enumerate(labels)}
    return labels, label2id


def load_tsv(path: Path) -> Tuple[List[str], List[int]]:
    """读取制表符分隔的（文本, 标签编号）数据集文件。

    每行格式为 ``文本\\t标签编号``，忽略空行与格式异常的行。
    """
    texts: List[str] = []
    labels: List[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        texts.append(parts[0])
        labels.append(int(parts[1]))
    return texts, labels


class IntentDataset(Dataset):
    """将（文本, 标签）列表封装为可供 DataLoader 使用的数据集。

    在构造阶段一次性完成 tokenize（不 padding），
    由 DataCollatorWithPadding 在取批次时动态补齐到批内最大长度，节省显存。
    """

    def __init__(self, texts: List[str], labels: List[int],
                 tokenizer: PreTrainedTokenizer, max_len: int) -> None:
        """初始化数据集。

        参数：
            texts: 原始查询文本列表。
            labels: 与 texts 一一对应的标签编号列表。
            tokenizer: 用于分词的 BERT 分词器。
            max_len: 单条文本的最大 token 数，超出截断。
        """
        # 批量分词比逐条分词快一个数量级，padding=False 由 collator 负责
        encodings = tokenizer(
            texts, truncation=True, max_length=max_len, padding=False,
        )
        self.input_ids = [torch.tensor(ids, dtype=torch.long) for ids in encodings["input_ids"]]
        self.attention_mask = [torch.tensor(mask, dtype=torch.long) for mask in encodings["attention_mask"]]
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        """返回第 idx 条样本的模型输入字典（含 labels）。"""
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }
