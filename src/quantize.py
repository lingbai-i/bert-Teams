"""动态量化：将微调后 BERT 的 Linear 层权重量化为 INT8。

动态量化（DQ）是训练后量化中最简单的方式：无需校准数据、无需重新训练，
推理时权重从 FP32 实时转 INT8 计算再转回。仅支持 CPU 推理。
预期效果：体积约为原来的 40%，CPU 推理明显加快，精度损失在 1 个百分点以内。

用法：
    python -m src.quantize --ckpt checkpoints/intent_bert/best.pt
"""
from pathlib import Path
from transformers import AutoTokenizer

import argparse
import json
import logging
import time

import torch

from src.config import CKPT_DIR, CLASS_FILE, DEV_FILE, MAX_LEN, MODEL_DIR
from src.data import build_dataloader, load_labels, load_txt
from src.model import load_classifier, state_dict_size_mb
from src.train import evaluate_accuracy

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def bench_latency(model, tokenizer, device: str, n: int = 100) -> float:
    """测量单条文本的平均推理延迟。

    参数:
        model: 待测模型。
        tokenizer: 分词器。
        device: 推理设备。
        n: 测量次数，取均值，默认 100。

    返回:
        平均延迟，单位毫秒。
    """
    enc = tokenizer("今天下午的会议改到几点了", return_tensors="pt",
                    truncation=True, max_length=MAX_LEN)
    enc = {k: v.to(device) for k, v in enc.items()}
    model.eval()
    with torch.no_grad():
        for _ in range(10):  # 预热，排除首次调用的初始化开销
            model(**enc)
        t0 = time.perf_counter()
        for _ in range(n):
            model(**enc)
    return (time.perf_counter() - t0) / n * 1000


def main(args) -> None:
    # 动态量化只支持 CPU：INT8 算子没有 CUDA 实现，放 GPU 会直接报错
    device = "cpu"
    label_names = load_labels(CLASS_FILE)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = load_classifier(MODEL_DIR, Path(args.ckpt), len(label_names), device)

    dev_t, dev_l = load_txt(DEV_FILE)
    dev_dl = build_dataloader(dev_t, dev_l, tokenizer, 64, MAX_LEN)

    acc_before = evaluate_accuracy(model, dev_dl, device)
    size_before = state_dict_size_mb(model)
    latency_before = bench_latency(model, tokenizer, device)
    logger.info(f"量化前: dev_acc={acc_before:.4f} 体积={size_before:.1f}MB "
                f"CPU延迟={latency_before:.1f}ms")

    quantized = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8)
    acc_after = evaluate_accuracy(quantized, dev_dl, device)
    latency_after = bench_latency(quantized, tokenizer, device)
    # 量化后 state_dict 含 INT8 张量，element_size 能正确反映体积
    size_after = state_dict_size_mb(quantized)
    logger.info(f"量化后: dev_acc={acc_after:.4f} 体积={size_after:.1f}MB "
                f"CPU延迟={latency_after:.1f}ms")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(quantized.state_dict(), out_dir / "best_int8.pt")
    (out_dir / "metrics.json").write_text(json.dumps({
        "dev_acc_before": acc_before, "dev_acc_after": acc_after,
        "size_mb_before": size_before, "size_mb_after": size_after,
        "cpu_latency_ms_before": latency_before,
        "cpu_latency_ms_after": latency_after,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"量化模型已保存到 {out_dir / 'best_int8.pt'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="BERT 动态量化（INT8，仅 CPU）")
    ap.add_argument("--ckpt", default=str(CKPT_DIR / "intent_bert" / "best.pt"))
    ap.add_argument("--output-dir", default=str(CKPT_DIR / "intent_bert_int8"))
    main(ap.parse_args())
