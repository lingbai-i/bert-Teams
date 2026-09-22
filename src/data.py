"""数据加载与预处理模块。

数据格式：train.txt / dev.txt / test.txt 每行为 `文本\t标签ID`，
标签名与 ID 的对应关系由 data/class.txt 的行序定义。
对外提供文本读取、标签读取、子采样与 DataLoader 构建四类接口，
被 train / evaluate / distill / prune 等模块共用。
"""
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from typing import List, Tuple

import random
import torch


def load_labels(class_file: Path) -> List[str]:
    """读取标签体系。

    参数:
        class_file: class.txt 路径，每行一个标签名，行号即标签 ID。

    返回:
        标签名列表，下标与标签 ID 一致。

    异常:
        FileNotFoundError: 文件不存在时抛出，调用方需先确认数据已就位。
    """
    # 标签名会作为 API 返回值与 jsonl 标签映射的键，必须去掉行尾换行符，
    # 否则 predict 返回 "policy_attendance\n"、evaluate 按名字建映射时全部查不到
    text = class_file.read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]


def load_txt(path: Path) -> Tuple[List[str], List[int]]:
    """读取 `文本\t标签ID` 格式的数据文件。

    参数:
        path: 数据文件路径，兼容 CRLF/LF 两种换行。

    返回:
        (文本列表, 标签 ID 列表)，两者等长；无标签分隔符的空行被跳过。
    """
    texts, labels = [], []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.rstrip("\r\n")
            if not line or "\t" not in line:
                continue

            text, label_str = line.rsplit("\t", 1)
            text = text.strip()

            # 跳过空文本
            if not text:
                continue

            # 标签非整数说明数据文件被污染，直接失败并给出定位信息；
            try:
                label = int(label_str)
            except ValueError as exc:
                raise ValueError(
                    f"{path} 第 {line_no} 行标签非法: {label_str!r}，请核对数据格式（文本\\t标签ID）"
                ) from exc

            texts.append(text)
            labels.append(label)

    return texts, labels


def subsample(
    texts: List[str], labels: List[int], n: int, seed: int
) -> Tuple[List[str], List[int]]:
    """有放回随机抽取 n 条（n 超过总量时返回全集的乱序副本）。

    参数:
        texts: 文本列表。
        labels: 标签 ID 列表。
        n: 抽取条数。
        seed: 随机种子，保证 mini 训练可复现。

    返回:
       长度均为 n；有放回抽样可能包含重复样本。
    """
    # 边界条件：如果原数据为空，直接返回空列表，避免报错
    if not texts:
        return [], []

    # 用局部 RNG 而非全局 random：既保证同 seed 可复现，也不改变其他模块的随机序列
    rng = random.Random(seed)
    # choices 是有放回抽样：同一样本可被多次抽出，返回长度恒为 n
    idx = rng.choices(range(len(texts)), k=max(n, 0))
    return [texts[i] for i in idx], [labels[i] for i in idx]


class IntentDataset(Dataset):
    """意图分类数据集，将文本编码为 BERT 输入张量。

    截断/padding 策略统一为 max_len，保证 batch 内张量形状一致。
    """

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_len: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, i: int) -> dict:
        enc = self.tokenizer(
            self.texts[i],
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[i], dtype=torch.long),
        }


def build_dataloader(
    texts: List[str],
    labels: List[int],
    tokenizer,
    batch_size: int,
    max_len: int,
    shuffle: bool = False,
) -> DataLoader:
    """由文本与标签列表构建 DataLoader。

    参数:
        texts: 文本列表。
        labels: 标签 ID 列表。
        tokenizer: HuggingFace tokenizer 实例。
        batch_size: 批次大小。
        max_len: 截断长度（token）。
        shuffle: 是否打乱（训练集传 True，评估传 False）。

    返回:
        迭代产出 {input_ids, attention_mask, label} 的 DataLoader。
    """
    ds = IntentDataset(texts, labels, tokenizer, max_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)



