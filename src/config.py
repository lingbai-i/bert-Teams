"""全局配置模块。

集中管理项目路径、标签文件位置与训练默认超参数。
所有模块统一从这里读取配置，禁止在业务代码中散落定义路径或超参数常量；
命令行参数优先级高于此处默认值。
"""
from pathlib import Path

# 项目根目录（src 的上一级）
ROOT = Path(__file__).resolve().parent.parent

# 数据与产物路径
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models" / "bert-base-chinese"
CKPT_DIR = ROOT / "checkpoints"
OUTPUT_DIR = ROOT / "outputs"
CLASS_FILE = DATA_DIR / "class.txt"
TRAIN_FILE = DATA_DIR / "train.txt"
DEV_FILE = DATA_DIR / "dev.txt"
TEST_FILE = DATA_DIR / "test.txt"
EVAL_FILE = DATA_DIR / "eval" / "test_queries.jsonl"
STOPWORD_FILE = DATA_DIR / "stopwords.txt"

# 权威模型权重：全量训练产物由组长确认后统一放在 models/ 下，评估环节只认这一份，
# 避免“每人一个 best.pt”导致指标无法对齐（见 README「模型产物规范」）。
# models/ 已被 .gitignore 排除，权重通过网盘/即时通讯分发，不进仓库。
BEST_CKPT = ROOT / "models" / "best.pt"

# badcase 归因分析（人工撰写）：文件存在时由 evaluate 追加到 badcases.txt 末尾，
# 这样重跑评估不会丢失结论，结论也不必硬编码进代码。
BADCASE_NOTES_FILE = DATA_DIR / "eval" / "badcase_notes.md"

# 训练默认超参数（单位与取值范围见各训练脚本参数说明）
MAX_LEN = 48        # 输入截断长度，单位 token；数据集文本较短，48 足够覆盖
BATCH_SIZE = 32     # 批次大小；8GB 显存在 max_len=48 下可承受
EPOCHS = 3          # 训练轮数
LR = 1e-4           # AdamW 初始学习率
SEED = 42           # 全局随机种子，保证实验可复现
