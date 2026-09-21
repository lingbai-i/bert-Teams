"""全局配置模块。

集中管理数据路径、模型名称与训练超参数，供训练、评估与推理脚本统一读取，
避免配置散落各处导致口径不一致。

依赖关系：
- 被 src/train.py、src/evaluate.py、src/infer.py 引用；
- 不依赖其他本地模块。
"""
from dataclasses import dataclass
from pathlib import Path

# 项目根目录（src 的上一级）
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Config:
    """意图识别任务的统一配置项。

    属性均为只读；训练、评估、推理脚本统一从本类读取配置。
    """

    # ---- 数据路径 ----
    data_dir: Path = PROJECT_ROOT / "data"
    class_file: Path = PROJECT_ROOT / "data" / "class.txt"
    train_file: Path = PROJECT_ROOT / "data" / "train.txt"
    dev_file: Path = PROJECT_ROOT / "data" / "dev.txt"
    test_file: Path = PROJECT_ROOT / "data" / "test.txt"
    eval_file: Path = PROJECT_ROOT / "data" / "eval" / "test_queries.jsonl"

    # ---- 模型 ----
    model_name: str = "bert-base-chinese"
    # 文本最大长度（token 数），超出部分截断
    max_len: int = 64
    # 类别总数，与 data/class.txt 行数一致
    num_labels: int = 9

    # ---- 推理阈值 ----
    # 所有类别的最大 softmax 概率低于该阈值时，判定为闲聊并拒绝回答
    reject_threshold: float = 0.5

    # ---- 训练超参数 ----
    seed: int = 42
    train_batch_size: int = 16
    eval_batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 2
    epochs: int = 2
    # 使用混合精度以适配 4GB 显存的消费级 GPU
    fp16: bool = True
    # 每隔多少训练步做一次验证与保存
    eval_steps: int = 2000
    logging_steps: int = 500

    # ---- 输出路径 ----
    model_dir: Path = PROJECT_ROOT / "models" / "bert-intent"
    report_dir: Path = PROJECT_ROOT / "outputs"
