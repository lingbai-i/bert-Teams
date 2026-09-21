"""fasttext 基线：词级别有监督文本分类。

fasttext 以词袋 + 分层 softmax 实现毫秒级分类，是深度学习路线中
最快的基线。输入格式为 `__label__标签名 词1 词2 ...`（jieba 分词）。

注意：fasttext 官方包没有 Windows + Python 3.12 的预编译产物，
需按 README「fasttext 环境说明」在 Python 3.11 环境中安装 fasttext-wheel 后运行。

用法：
    python -m src.fasttext_baseline                 # 默认参数训练
    python -m src.fasttext_baseline --autotune 300  # 自动调参（300 秒）
"""
from pathlib import Path
from typing import List

import argparse
import json
import logging
import time

import jieba

from src.config import CKPT_DIR, CLASS_FILE, DEV_FILE, OUTPUT_DIR, TEST_FILE, TRAIN_FILE
from src.data import load_labels, load_txt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

try:
    import fasttext
except ImportError as e:
    raise SystemExit(
        "未安装 fasttext。请先确认使用 Python 3.11 环境并安装依赖：\n"
        "  uv pip install -r requirements.txt"
    ) from e


def to_fasttext_lines(texts: List[str], labels: List[int],
                      label_names: List[str]) -> List[str]:
    """把样本转为 fasttext 有监督格式 `__label__标签名 词1 词2 ...`。"""
    return [f"__label__{label_names[l]} " + " ".join(w for w in jieba.lcut(t) if w.strip())
            for t, l in zip(texts, labels)]


def write_lines(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(args) -> None:
    label_names = load_labels(CLASS_FILE)
    # fasttext 输入文件为生成物，放 outputs/ 下，不进版本库
    proc_dir = OUTPUT_DIR / "fasttext"
    for split, src in [("train", TRAIN_FILE), ("dev", DEV_FILE), ("test", TEST_FILE)]:
        texts, labels = load_txt(src)
        write_lines(proc_dir / f"{split}_words.txt",
                    to_fasttext_lines(texts, labels, label_names))
    logger.info(f"fasttext 格式数据已生成到 {proc_dir}")

    t0 = time.time()
    if args.autotune > 0:
        model = fasttext.train_supervised(
            input=str(proc_dir / "train_words.txt"),
            autotuneValidationFile=str(proc_dir / "dev_words.txt"),
            autotuneDuration=args.autotune, verbose=3)
    else:
        model = fasttext.train_supervised(
            input=str(proc_dir / "train_words.txt"),
            epoch=args.epochs, lr=args.lr, wordNgrams=2, verbose=2)
    logger.info(f"训练完成，用时 {time.time() - t0:.0f}s")

    n, precision, recall = model.test(str(proc_dir / "test_words.txt"))
    # fasttext 单标签场景下 P@1 即准确率
    logger.info(f"test 集: n={n} acc={precision:.4f}")

    sample = "今天下午的会议改到几点了"
    words = " ".join(jieba.lcut(sample))
    model.predict(words)  # 预热
    t0 = time.perf_counter()
    for _ in range(100):
        model.predict(words)
    latency = (time.perf_counter() - t0) / 100 * 1000

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out_dir / "ft.bin"))
    (out_dir / "metrics.json").write_text(json.dumps({
        "args": vars(args),
        "test_acc": precision, "latency_ms": latency,
        "model_file_mb": (out_dir / "ft.bin").stat().st_size / 1024 / 1024,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"模型已保存到 {out_dir / 'ft.bin'}，单条延迟 {latency:.2f}ms")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="fasttext 词级别分类基线")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--autotune", type=int, default=0,
                    help="自动调参时长（秒），0 表示使用固定超参数")
    ap.add_argument("--output-dir", default=str(CKPT_DIR / "fasttext"))
    main(ap.parse_args())
