"""模型构建模块。

负责创建 BERT 序列分类模型、从检查点恢复模型，以及蒸馏用的 BiLSTM 学生模型。
学生模型复用 BERT 的分词结果（input_ids），但用独立的 Embedding + BiLSTM
提取特征，参数量从教师的 102M 降到约 25M，便于在 CPU 上毫秒级推理。
"""
from pathlib import Path
from transformers import AutoModelForSequenceClassification

import torch


def build_classifier(model_dir: Path, num_labels: int, device: str):
    """从本地预训练目录创建 BERT 分类模型。

    参数:
        model_dir: bert-base-chinese 本地目录。
        num_labels: 类别数，由 data/class.txt 行数决定。
        device: "cuda" 或 "cpu"。

    返回:
        已切换到指定设备的 AutoModelForSequenceClassification。
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir), num_labels=num_labels
    )
    return model.to(device)


def load_classifier(model_dir: Path, ckpt_path: Path, num_labels: int, device: str):
    """创建 BERT 分类模型并加载微调后的权重。

    参数:
        model_dir: bert-base-chinese 本地目录。
        ckpt_path: 训练保存的 state_dict（.pt）路径。
        num_labels: 类别数。
        device: "cuda" 或 "cpu"。

    返回:
        加载权重后处于 eval 模式的模型。

    异常:
        FileNotFoundError: ckpt_path 不存在时抛出，提示先完成训练。
    """
    if not ckpt_path.exists():
        raise FileNotFoundError(f"未找到模型权重：{ckpt_path}，请先完成训练")
    model = build_classifier(model_dir, num_labels, device)
    state = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


class BiLSTMClassifier(torch.nn.Module):
    """蒸馏用 BiLSTM 学生模型。

    输入复用 BERT tokenizer 的编码结果（input_ids + attention_mask），
    经过独立 Embedding、双向 LSTM、有效位平均池化与线性分类头输出 logits。
    对 [CLS]/[SEP]/[PAD] 位置的向量做掩码置零，避免特殊符号干扰句向量。
    """

    def __init__(self, vocab_size: int, num_labels: int, embed_size: int = 128,
                 hidden_size: int = 256, num_layers: int = 3, dropout: float = 0.3):
        """初始化学生模型。

        参数:
            vocab_size: 词表大小，与 BERT 分词器一致（21128）。
            num_labels: 类别数。
            embed_size: 词向量维度。
            hidden_size: LSTM 单向隐藏层维度，双向拼接后为 2 倍。
            num_layers: LSTM 堆叠层数。
            dropout: 池化后的随机失活概率。
        """
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_size)
        self.bilstm = torch.nn.LSTM(
            input_size=embed_size, hidden_size=hidden_size,
            num_layers=num_layers, bidirectional=True, batch_first=True)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.classifier = torch.nn.Linear(hidden_size * 2, num_labels)

    def forward(self, input_ids, attention_mask):
        """前向传播。

        参数:
            input_ids: [batch, seq_len] 的 token id 张量。
            attention_mask: [batch, seq_len] 的注意力掩码（1 有效，0 为 PAD）。

        返回:
            [batch, num_labels] 的 logits。
        """
        embed = self.embedding(input_ids)
        # [CLS]=101、[SEP]=102 与 PAD 位不参与句向量计算
        valid_mask = ((input_ids != 101) & (input_ids != 102)).long() * attention_mask
        valid_mask = valid_mask.unsqueeze(-1)
        lstm_out, _ = self.bilstm(embed * valid_mask)
        summed = (lstm_out * valid_mask).sum(dim=1)
        token_cnt = valid_mask.sum(dim=1).clamp(min=1)
        pooled = self.dropout(summed / token_cnt)
        return self.classifier(pooled)


def count_parameters(model) -> int:
    """统计模型可训练参数量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def state_dict_size_mb(model) -> float:
    """估算模型 state_dict 序列化后的体积（MB，float32 按 4 字节计）。"""
    total_bytes = sum(p.numel() * p.element_size() for p in model.state_dict().values())
    return total_bytes / 1024 / 1024
