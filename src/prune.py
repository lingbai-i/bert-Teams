"""全局非结构化剪枝：按比例置零 BERT 各层注意力 query 权重中绝对值最小的参数。

非结构化剪枝属于「逻辑剪枝」：权重被置零但张量形状不变，因此模型体积不变、
CPU 推理提速有限；其价值在于验证网络中存在大量冗余参数（精度几乎不掉），
配合稀疏格式存储或专用推理库才能进一步带来体积/速度收益。

用法：
    python -m src.prune --ckpt checkpoints/intent_bert/best.pt --amount 0.3
"""
from pathlib import Path
from torch.nn.utils import prune
from transformers import AutoTokenizer

import argparse
import json
import logging

import torch

from src.config import CKPT_DIR, CLASS_FILE, DEV_FILE, MAX_LEN, MODEL_DIR
from src.data import build_dataloader, load_labels, load_txt
from src.model import load_classifier
from src.train import evaluate_accuracy

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def query_sparsity(model) -> float:
    """计算各 encoder 层注意力 query 权重的整体稀疏度（零值占比）。"""
    total = zeros = 0
    for layer in model.bert.encoder.layer:
        weight = layer.attention.self.query.weight
        total += weight.numel()
        zeros += (weight == 0).sum().item()
    return zeros / total if total else 0.0


def main(args) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    label_names = load_labels(CLASS_FILE)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = load_classifier(MODEL_DIR, Path(args.ckpt), len(label_names), device)

    dev_t, dev_l = load_txt(DEV_FILE)
    dev_dl = build_dataloader(dev_t, dev_l, tokenizer, 64, MAX_LEN)

    acc_before = evaluate_accuracy(model, dev_dl, device)
    sparsity_before = query_sparsity(model)
    logger.info(f"剪枝前: dev_acc={acc_before:.4f} 稀疏度={sparsity_before:.2%}")

    # 对所有 encoder 层的 query 权重做全局 L1 非结构化剪枝
    params = [(layer.attention.self.query, "weight")
              for layer in model.bert.encoder.layer]
    prune.global_unstructured(params, pruning_method=prune.L1Unstructured,
                              amount=args.amount)
    # 固化剪枝：把掩码合并进权重，移除重参数化钩子
    for module, name in params:
        prune.remove(module, name)

    acc_after = evaluate_accuracy(model, dev_dl, device)
    sparsity_after = query_sparsity(model)
    logger.info(f"剪枝后: dev_acc={acc_after:.4f} 稀疏度={sparsity_after:.2%}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "best.pt")
    (out_dir / "metrics.json").write_text(json.dumps({
        "amount": args.amount,
        "dev_acc_before": acc_before, "dev_acc_after": acc_after,
        "sparsity_before": sparsity_before, "sparsity_after": sparsity_after,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"剪枝模型已保存到 {out_dir / 'best.pt'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="BERT 全局非结构化剪枝")
    ap.add_argument("--ckpt", default=str(CKPT_DIR / "intent_bert" / "best.pt"))
    ap.add_argument("--amount", type=float, default=0.3,
                    help="全局置零比例，取值 (0, 1)")
    ap.add_argument("--output-dir", default=str(CKPT_DIR / "intent_bert_pruned"))
    main(ap.parse_args())
