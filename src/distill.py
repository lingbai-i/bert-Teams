"""知识蒸馏：将教师模型（微调后 BERT）的知识迁移到 BiLSTM 学生模型。

软标签蒸馏。总损失 loss = alpha * T^2 * KL(学生/T, 教师/T) + (1 - alpha) * 硬标签损失，
其中 T 为温度系数（一般取 2~5），alpha 控制软标签权重（一般取 0.5~0.9），
T^2 用于平衡温度升高后软标签梯度尺度被稀释的问题。
硬标签默认取教师预测的类别（教师兜底纠错），也可用 --hard-source label 换为人工标注。

用法：
    python -m src.distill --teacher checkpoints/intent_bert/best.pt
    python -m src.distill --subsample 5000 --epochs 1   # 快速验证流程
"""
from pathlib import Path
from torch.optim import AdamW
from transformers import AutoTokenizer

import argparse
import json
import logging
import time

import torch
import torch.nn.functional as F

from src.config import (
    BATCH_SIZE, CKPT_DIR, CLASS_FILE, DEV_FILE, MAX_LEN, MODEL_DIR,
    SEED, TRAIN_FILE,
)
from src.data import build_dataloader, load_labels, load_txt, subsample
from src.model import BiLSTMClassifier, count_parameters, load_classifier
from src.train import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def evaluate_bilstm(model, dl, device: str) -> float:
    """在验证集上评估 BiLSTM 学生模型的准确率。"""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in dl:
            logits = model(batch["input_ids"].to(device),
                           batch["attention_mask"].to(device))
            correct += (logits.argmax(-1).cpu() == batch["label"]).sum().item()
            total += len(batch["label"])
    return correct / max(total, 1)


def main(args) -> None:
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    label_names = load_labels(CLASS_FILE)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    vocab_size = tokenizer.vocab_size

    teacher = load_classifier(MODEL_DIR, Path(args.teacher), len(label_names), device)
    for p in teacher.parameters():
        p.requires_grad_(False)
    student = BiLSTMClassifier(vocab_size, len(label_names),
                               embed_size=args.embed_size,
                               hidden_size=args.hidden_size).to(device)
    logger.info(f"教师参数量={count_parameters(teacher) / 1e6:.1f}M "
                f"学生参数量={count_parameters(student) / 1e6:.1f}M")

    train_t, train_l = load_txt(TRAIN_FILE)
    dev_t, dev_l = load_txt(DEV_FILE)
    if args.subsample:
        train_t, train_l = subsample(train_t, train_l, args.subsample, SEED)
    train_dl = build_dataloader(train_t, train_l, tokenizer, args.bs, MAX_LEN, shuffle=True)
    dev_dl = build_dataloader(dev_t, dev_l, tokenizer, args.bs, MAX_LEN)
    logger.info(f"蒸馏训练集={len(train_t)}，硬标签来源={args.hard_source}")

    opt = AdamW(student.parameters(), lr=args.lr)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_dev_acc = 0.0
    for ep in range(args.epochs):
        student.train()
        t0 = time.time()
        total_loss = 0.0
        for batch in train_dl:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            true_labels = batch["label"].to(device)
            opt.zero_grad()
            with torch.no_grad():
                teacher_logits = teacher(input_ids=input_ids,
                                         attention_mask=mask).logits
            student_logits = student(input_ids, mask)

            # 软标签损失：KL(学生/T, 教师/T)，log_target 形式数值更稳定
            soft_loss = F.kl_div(
                F.log_softmax(student_logits / args.temperature, dim=-1),
                F.log_softmax(teacher_logits / args.temperature, dim=-1),
                reduction="batchmean", log_target=True,
            ) * args.temperature * args.temperature
            # 硬标签损失：默认以教师预测类别为目标，也可切换为人工标注
            hard_target = (teacher_logits.argmax(-1) if args.hard_source == "teacher"
                           else true_labels)
            hard_loss = F.cross_entropy(student_logits, hard_target)
            loss = args.alpha * soft_loss + (1 - args.alpha) * hard_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()

        dev_acc = evaluate_bilstm(student, dev_dl, device)
        logger.info(f"epoch {ep + 1}/{args.epochs} loss={total_loss / len(train_dl):.4f} "
                    f"dev_acc={dev_acc:.4f} 用时={time.time() - t0:.0f}s")
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            torch.save(student.state_dict(), out_dir / "best.pt")

    (out_dir / "metrics.json").write_text(json.dumps({
        "args": vars(args),
        "best_dev_acc": best_dev_acc,
        "student_params": count_parameters(student),
        "student_vocab_size": vocab_size,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"蒸馏完成 best_dev_acc={best_dev_acc:.4f}，权重已保存到 {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="BERT -> BiLSTM 软标签知识蒸馏")
    ap.add_argument("--teacher", default=str(CKPT_DIR / "intent_bert" / "best.pt"),
                    help="教师模型权重路径")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=1e-3, help="学生模型学习率")
    ap.add_argument("--temperature", type=float, default=2.0, help="温度系数 T")
    ap.add_argument("--alpha", type=float, default=0.7, help="软标签损失权重")
    ap.add_argument("--embed-size", type=int, default=128)
    ap.add_argument("--hidden-size", type=int, default=256)
    ap.add_argument("--hard-source", choices=["teacher", "label"], default="teacher",
                    help="硬标签来源：teacher 为教师预测类别，label 为人工标注")
    ap.add_argument("--subsample", type=int, default=0, help="训练子采样条数，0 为全量")
    ap.add_argument("--output-dir", default=str(CKPT_DIR / "intent_bilstm_distill"))
    main(ap.parse_args())
