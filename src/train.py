"""基线训练入口：BERT 意图分类器微调。

用法：
    python -m src.train                          # 全量训练（默认参数）
    python -m src.train --subsample 5000 --epochs 1   # mini 训练，验证流程

训练流程：加载数据 -> 可选子采样 -> AdamW + 线性 warmup -> bf16 混合精度训练，
每个 epoch 在 dev 集上评估并保存最优权重，结束后在 test 集上给出最终准确率，
训练记录写入输出目录的 metrics.json。
"""
from pathlib import Path
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

import argparse
import json
import logging
import random
import time

import numpy as np
import torch

from src.config import (
    BATCH_SIZE, CKPT_DIR, DEV_FILE, EPOCHS, LR, MAX_LEN, MODEL_DIR,
    SEED, TEST_FILE, TRAIN_FILE,
)
from src.data import build_dataloader, load_labels, load_txt, subsample
from src.model import build_classifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """固定全部随机源，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate_accuracy(model, dl: DataLoader, device: str) -> float:
    """在给定 DataLoader 上计算分类准确率。

    参数:
        model: 处于任意模式的分类模型（函数内切换为 eval）。
        dl: 评估数据加载器。
        device: 推理设备。

    返回:
        准确率，取值 [0, 1]；空数据集返回 0.0。
    """
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in dl:
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            ).logits
            correct += (logits.argmax(-1).cpu() == batch["label"]).sum().item()
            total += len(batch["label"])
    return correct / max(total, 1)


def main(args) -> None:
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    labels = load_labels(Path(args.class_file)) if args.class_file else load_labels(
        __import__("src.config", fromlist=["CLASS_FILE"]).CLASS_FILE)
    num_labels = len(labels)
    logger.info(f"设备: {device}，类别数: {num_labels}")

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = build_classifier(MODEL_DIR, num_labels, device)

    train_t, train_l = load_txt(TRAIN_FILE)
    dev_t, dev_l = load_txt(DEV_FILE)
    test_t, test_l = load_txt(TEST_FILE)
    if args.subsample:
        train_t, train_l = subsample(train_t, train_l, args.subsample, SEED)
    logger.info(f"train={len(train_t)} dev={len(dev_t)} test={len(test_t)}")

    train_dl = build_dataloader(train_t, train_l, tokenizer, args.bs, MAX_LEN, shuffle=True)
    dev_dl = build_dataloader(dev_t, dev_l, tokenizer, args.bs, MAX_LEN)
    test_dl = build_dataloader(test_t, test_l, tokenizer, args.bs, MAX_LEN)

    opt = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_dl) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total_steps), total_steps)
    # bf16 在 40 系显卡上无需 GradScaler 的梯度缩放即可稳定训练
    use_amp = device == "cuda"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_dev_acc = 0.0
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        for step, batch in enumerate(train_dl):
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                out = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=batch["label"].to(device),
                )
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total_loss += out.loss.item()
            if step % 200 == 0:
                logger.info(f"epoch {ep + 1} step {step}/{len(train_dl)} loss={out.loss.item():.4f}")

        dev_acc = evaluate_accuracy(model, dev_dl, device)
        record = {"epoch": ep + 1, "train_loss": total_loss / len(train_dl),
                  "dev_acc": dev_acc, "seconds": round(time.time() - t0, 1)}
        history.append(record)
        logger.info(f"epoch {ep + 1}/{args.epochs} 完成: {record}")
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            torch.save(model.state_dict(), out_dir / "best.pt")
            logger.info(f"保存当前最优权重（dev_acc={dev_acc:.4f}）")

    # 用 dev 最优权重在 test 集上给出最终指标
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device,
                                     weights_only=True))
    test_acc = evaluate_accuracy(model, test_dl, device)
    metrics = {
        "args": vars(args),
        "best_dev_acc": best_dev_acc,
        "test_acc": test_acc,
        "history": history,
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"训练完成 best_dev_acc={best_dev_acc:.4f} test_acc={test_acc:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="BERT 意图分类基线训练")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--bs", type=int, default=BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--subsample", type=int, default=0,
                    help="训练集子采样条数，0 表示全量；mini 训练建议 5000")
    ap.add_argument("--output-dir", default=str(CKPT_DIR / "intent_bert"))
    ap.add_argument("--class-file", default="")
    main(ap.parse_args())
