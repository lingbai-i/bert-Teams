"""模型推理封装模块。

加载训练好的意图分类模型，提供文本的意图预测与"低置信度拒答"判定，
供评估脚本与推理 CLI 复用同一套推理逻辑，保证线上/线下口径一致。

依赖关系：
- 依赖 src.config、src.dataset；
- 被 src/evaluate.py、src/infer.py 引用。
"""
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding)

from src.config import Config
from src.dataset import IntentDataset, load_class_labels


class IntentPredictor:
    """意图分类模型推理器。

    根据 softmax 最高概率与拒答阈值决定最终行为：
    - 最高概率低于阈值：所有类别可能性都低，判定为闲聊，拒绝提供业务回答；
    - 最高概率不低于阈值且预测类别为 other_chitchat：判定为闲聊，拒绝提供业务回答；
    - 其余情况：返回预测的业务意图，可以回答。
    """

    def __init__(self, model_dir: Path | None = None, threshold: float | None = None) -> None:
        """初始化推理器。

        参数：
            model_dir: 模型权重目录，默认使用配置中的模型输出目录。
            threshold: 拒答阈值，默认使用配置中的 reject_threshold。
        """
        cfg = Config()
        self.cfg = cfg
        model_dir = model_dir or cfg.model_dir
        self.threshold = cfg.reject_threshold if threshold is None else threshold

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        self.model.eval()
        # 无 CUDA 时自动退回 CPU，训练/推理均保持与训练时一致的设备选择
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.id2label, _ = load_class_labels(cfg.class_file)

    def predict_probs(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """批量预测文本的 softmax 概率矩阵，形状为 (N, num_labels)。

        参数：
            texts: 待预测文本列表。
            batch_size: 推理批次大小。

        返回：
            每行对应一条文本在各类别上的概率分布。
        """
        dummy_labels = [0] * len(texts)
        dataset = IntentDataset(texts, dummy_labels, self.tokenizer, self.cfg.max_len)
        collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        loader = DataLoader(
            dataset, batch_size=batch_size, collate_fn=collator, shuffle=False,
        )

        all_probs: List[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items() if k != "labels"}
                logits = self.model(**batch).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                all_probs.append(probs)
        return np.concatenate(all_probs, axis=0)

    def predict(self, text: str) -> Dict[str, object]:
        """预测单条文本，返回意图、置信度与是否拒答的判定结果。

        参数：
            text: 用户查询文本。

        返回：
            包含 text / intent / confidence / rejected / reason 的字典。
        """
        probs = self.predict_probs([text])[0]
        max_prob = float(probs.max())
        pred_id = int(probs.argmax())
        pred_label = self.id2label[pred_id]

        if max_prob < self.threshold:
            return {
                "text": text,
                "intent": "other_chitchat（闲聊）",
                "confidence": max_prob,
                "rejected": True,
                "reason": f"所有类别置信度均低于阈值 {self.threshold}，判定为闲聊，拒绝回答",
            }
        if pred_label == "other_chitchat":
            return {
                "text": text,
                "intent": pred_label,
                "confidence": max_prob,
                "rejected": True,
                "reason": "预测意图为闲聊类，不提供业务回答",
            }
        return {
            "text": text,
            "intent": pred_label,
            "confidence": max_prob,
            "rejected": False,
            "reason": None,
        }
