"""训练脚本：在 train.txt 上微调 bert-base-chinese 意图分类模型。

训练过程中以 dev.txt 计算准确率与宏平均 F1，按验证指标保存最优 checkpoint；
训练结束后将最优模型与标签映射保存至 models/bert-intent，
并将训练日志（loss/acc 序列）导出为 outputs/train_log.json 供报告使用。

依赖关系：
- 依赖 src.config、src.dataset；
- 通过 `python -m src.train` 运行。
"""
import json
import math
from pathlib import Path

import numpy as np
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments,
                          set_seed)
from sklearn.metrics import accuracy_score, f1_score

from src.config import Config
from src.dataset import IntentDataset, load_class_labels, load_tsv


def compute_metrics(eval_pred) -> dict:
    """计算验证集指标：整体准确率与宏平均 F1。"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
    }


def main() -> None:
    """执行训练主流程。"""
    cfg = Config()
    cfg.report_dir.mkdir(parents=True, exist_ok=True)
    cfg.model_dir.mkdir(parents=True, exist_ok=True)

    set_seed(cfg.seed)

    # ---- 加载数据与类别映射 ----
    id2label, label2id = load_class_labels(cfg.class_file)
    if len(id2label) != cfg.num_labels:
        raise ValueError(
            f"class.txt 类别数 {len(id2label)} 与配置 num_labels={cfg.num_labels} 不一致",
        )
    train_texts, train_labels = load_tsv(cfg.train_file)
    dev_texts, dev_labels = load_tsv(cfg.dev_file)
    print(f"训练集 {len(train_texts)} 条，验证集 {len(dev_texts)} 条，类别数 {len(id2label)}")

    # ---- 加载预训练模型与分词器 ----
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=cfg.num_labels,
        id2label={str(i): name for i, name in enumerate(id2label)},
        label2id=label2id,
    )

    train_dataset = IntentDataset(train_texts, train_labels, tokenizer, cfg.max_len)
    dev_dataset = IntentDataset(dev_texts, dev_labels, tokenizer, cfg.max_len)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # transformers 5.x 仅支持 warmup_steps，此处由 warmup_ratio 换算：
    # 总优化步数 = 每轮批次数（含梯度累积）× 训练轮数
    steps_per_epoch = math.ceil(len(train_dataset) / (cfg.train_batch_size * cfg.gradient_accumulation_steps))
    warmup_steps = int(steps_per_epoch * cfg.epochs * cfg.warmup_ratio)

    training_args = TrainingArguments(
        output_dir=str(cfg.report_dir / "checkpoints"),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.train_batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_steps=warmup_steps,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        fp16=cfg.fp16,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.eval_steps,
        logging_steps=cfg.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=2,
        seed=cfg.seed,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # ---- 保存最优模型与附属文件 ----
    trainer.save_model(str(cfg.model_dir))
    tokenizer.save_pretrained(str(cfg.model_dir))
    label_map = {"id2label": {str(i): name for i, name in enumerate(id2label)}, "label2id": label2id}
    (cfg.model_dir / "label_map.json").write_text(
        json.dumps(label_map, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    # 导出训练日志供报告绘制曲线
    (cfg.report_dir / "train_log.json").write_text(
        json.dumps(trainer.state.log_history, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    dev_metrics = trainer.evaluate()
    print("验证集最终指标:", dev_metrics)
    print(f"最优模型已保存至 {cfg.model_dir}")


if __name__ == "__main__":
    main()
