"""知识蒸馏：将教师模型（微调后 BERT）的知识迁移到 BiLSTM 学生模型。

软标签蒸馏。总损失 loss = alpha * T^2 * KL(学生/T, 教师/T) + (1 - alpha) * 硬标签损失，
其中 T 为温度系数（一般取 2~5），alpha 控制软标签权重（一般取 0.5~0.9），
T^2 用于平衡温度升高后软标签梯度尺度被稀释的问题。
硬标签默认取教师预测的类别（教师兜底纠错），也可用 --hard-source label 换为人工标注。

实测结论（本项目模板合成数据，教师为 202500 条全量训练、seed=42）：
    体积：390.2 MB -> 25.4 MB，为教师的 6.5%，与「约 25MB」的预期一致。
    参数量：102.3M -> 6.7M（6.5%）。
    精度：学生准确率 0.9847、宏平均 F1 0.9848，与教师的 0.9844 / 0.9846 持平甚至略高。
        学生反超的可能原因：参数量少、在模板化数据上过拟合更轻；
        软标签的平滑分布也起了正则作用。
    延迟：CPU 单条 20.1ms，约为教师 85.8ms 的四分之一。
    即蒸馏是三种压缩方式中唯一同时显著省体积又提速的方法。

训练与评估的数据划分（刻意分开）：
    data/dev.txt 用于每个 epoch 后的最优权重选择，data/test.txt 用于最终指标报告。
    这样划分是为了避免用测试集选模型造成信息泄漏。
    已知问题：data/dev.txt 只有 8 个类别（缺 other_chitchat），
    因此最优权重的选择信号是不完整的，这只影响选到哪一轮的权重，
    不影响最终在 test 上的指标口径（test 是完整 9 类）。

延迟口径：训练与评估用 GPU，但延迟统一搬到 CPU 上测——
量化后的 INT8 算子只有 CPU 实现，三件套必须在同一设备口径下比较。

用法：
    python -m src.distill --teacher checkpoints/intent_bert/best.pt
    python -m src.distill --subsample 5000 --epochs 1   # 快速验证流程
"""
from pathlib import Path
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score)
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from typing import Any, Dict, List, Tuple

import argparse
import json
import logging
import os
import tempfile
import time

import torch
import torch.nn.functional as F

from src.config import (
    BATCH_SIZE, CKPT_DIR, CLASS_FILE, DEV_FILE, MAX_LEN, MODEL_DIR, SEED,
    TEST_FILE, TRAIN_FILE,
)
from src.data import build_dataloader, load_labels, load_txt, subsample
from src.evaluate import load_any, predict_all
from src.model import BiLSTMClassifier, count_parameters, load_classifier
from src.train import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def evaluate_bilstm(model, dl: DataLoader, device: str) -> float:
    """在验证集上评估 BiLSTM 学生模型的准确率。

    参数:
        model: 学生模型。
        dl: 验证集加载器。
        device: 推理设备。

    返回:
        准确率，取值 [0, 1]；数据集为空时返回 0。

    说明:
        只算准确率，用于每个 epoch 后的最优权重选择。
        最终对外报告的指标由 evaluate_student 计算，含宏平均三指标。
    """
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in dl:
            logits = model(batch["input_ids"].to(device),
                           batch["attention_mask"].to(device))
            correct += (logits.argmax(-1).cpu() == batch["label"]).sum().item()
            total += len(batch["label"])
    return correct / max(total, 1)


def predict_bilstm_all(model, dl: DataLoader,
                       device: str) -> Tuple[List[int], List[int]]:
    """遍历 DataLoader 收集学生模型的预测与真实标签。

    参数:
        model: 学生模型。
        dl: 数据加载器。
        device: 推理设备。

    返回:
        (预测标签 ID 列表, 真实标签 ID 列表)，两者等长且同序。

    说明:
        不能复用 src.evaluate.predict_all：那个函数按
        model(input_ids=..., attention_mask=...).logits 调用，
        而 BiLSTMClassifier 的 forward 直接返回 logits 张量、没有 logits 属性。
        因此这里单独实现一份，只取返回值的张量本身。
    """
    predictions: List[int] = []
    golds: List[int] = []
    model.eval()
    with torch.no_grad():
        for batch in dl:
            logits = model(batch["input_ids"].to(device),
                           batch["attention_mask"].to(device))
            predictions.extend(logits.argmax(-1).cpu().tolist())
            golds.extend(batch["label"].tolist())
    return predictions, golds


def metrics_from_predictions(predictions: List[int], gold_labels: List[int],
                             label_names: List[str]) -> Dict[str, Any]:
    """由预测与真实标签计算准确率与宏平均 precision/recall/F1。

    参数:
        predictions: 预测标签 ID 列表。
        gold_labels: 真实标签 ID 列表，与 predictions 等长。
        label_names: 标签名列表，下标即标签 ID。

    返回:
        含 sample_count、accuracy、macro_precision、macro_recall、macro_f1、
        report 六个键的字典。

    说明:
        显式传入 labels=range(类别数)，把标签空间固定为全部类别。
        若不传，当某个类别在数据中缺席时 sklearn 会抛
        「Number of classes does not match size of target_names」——
        data/dev.txt 缺一个类，正是这种情形。
        显式传 labels 后，缺席类别会以 support=0 的形式出现在报告里。
    """
    label_ids = list(range(len(label_names)))
    return {
        "sample_count": len(gold_labels),
        "accuracy": float(accuracy_score(gold_labels, predictions)),
        "macro_precision": float(precision_score(
            gold_labels, predictions, labels=label_ids, average="macro",
            zero_division=0)),
        "macro_recall": float(recall_score(
            gold_labels, predictions, labels=label_ids, average="macro",
            zero_division=0)),
        "macro_f1": float(f1_score(
            gold_labels, predictions, labels=label_ids, average="macro",
            zero_division=0)),
        "report": classification_report(
            gold_labels, predictions, labels=label_ids,
            target_names=label_names, digits=4, zero_division=0),
    }


def measure_size_mb(model) -> float:
    """测量模型序列化后的真实体积。

    参数:
        model: 待测量模型（教师或学生均可）。

    返回:
        体积，单位 MB。

    说明:
        做法是把 state_dict 写进系统临时文件、取字节数、再删除，
        因此不污染任何输出目录。

        不用 src.model.state_dict_size_mb 的原因：该函数逐张量求和，
        对量化模型会漏掉 packed params。本模块虽不做量化，
        但为了让三件套的体积口径完全一致（对比表要求），统一用同一算法。

        副作用：会经历一次完整的磁盘序列化。
    """
    handle, temp_name = tempfile.mkstemp(suffix=".pt")
    # Windows 上文件被占用时 torch.save 无法写入，必须先关掉 mkstemp 返回的句柄
    os.close(handle)
    temp_path = Path(temp_name)
    try:
        torch.save(model.state_dict(), temp_path)
        return temp_path.stat().st_size / 1024 / 1024
    finally:
        # 放在 finally 里：即使上面抛异常也要删掉临时文件
        temp_path.unlink(missing_ok=True)


def bench_latency_bert(model, tokenizer, device: str, n: int = 100) -> float:
    """测量 BERT 系列模型单条文本的平均推理延迟（毫秒）。

    参数:
        model: BERT 系列模型。
        tokenizer: 分词器。
        device: 推理设备。三件套统一传 "cpu" 以保证可比性。
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


def bench_latency_bilstm(model, tokenizer, device: str,
                         n: int = 100) -> float:
    """测量 BiLSTM 学生模型单条文本的平均推理延迟（毫秒）。

    参数:
        model: 学生模型。
        tokenizer: 分词器。
        device: 推理设备。
        n: 测量次数，取均值，默认 100。

    返回:
        平均延迟，单位毫秒。

    说明:
        与 bench_latency_bert 分开实现而不是加一个布尔开关：
        两类模型的调用签名不同——学生只接受 input_ids 与 attention_mask，
        不接受 token_type_ids，传 **enc 会直接报参数错误。
        写成两个函数能让签名差异在代码里直接可见。
    """
    enc = tokenizer("今天下午的会议改到几点了", return_tensors="pt",
                    truncation=True, max_length=MAX_LEN)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    model.eval()
    with torch.no_grad():
        for _ in range(10):  # 预热
            model(input_ids, attention_mask)
        t0 = time.perf_counter()
        for _ in range(n):
            model(input_ids, attention_mask)
    return (time.perf_counter() - t0) / n * 1000


def verify_student_artifact(ckpt_path: Path, vocab_size: int, num_labels: int,
                            tokenizer, sample_text: str) -> int:
    """验证保存的学生权重能否被独立加载并推理。

    参数:
        ckpt_path: 学生权重文件路径。
        vocab_size: 词表大小，须与训练时一致，否则 Embedding 形状对不上。
        num_labels: 类别数。
        tokenizer: 分词器。
        sample_text: 用于验证推理链路的文本。

    返回:
        预测出的标签 ID。

    异常:
        FileNotFoundError: 学生权重文件不存在时抛出。

    说明:
        这里从零新建一个学生模型再加载权重，验证的是「产物可独立使用」——
        不依赖训练过程中的任何内存状态。部署层也按同样方式加载。
    """
    if not ckpt_path.exists():
        raise FileNotFoundError(f"未找到学生权重：{ckpt_path}")
    probe = BiLSTMClassifier(vocab_size, num_labels)
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    probe.load_state_dict(state)
    probe.to("cpu").eval()
    enc = tokenizer(sample_text, return_tensors="pt", truncation=True,
                    max_length=MAX_LEN)
    with torch.no_grad():
        predicted = int(probe(enc["input_ids"],
                              enc["attention_mask"]).argmax(-1)[0])
    del probe
    return predicted


def main(args) -> None:
    """执行蒸馏训练、评估、保存与产物验证的完整流程。

    参数:
        args: 命令行参数，含 teacher、epochs、bs、lr、temperature、alpha、
            embed_size、hidden_size、hard_source、subsample、input、output_dir。

    返回:
        None。

    异常:
        FileNotFoundError: 教师权重不存在时由 load_classifier 抛出。
    """
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    label_names = load_labels(CLASS_FILE)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    vocab_size = tokenizer.vocab_size

    teacher = load_classifier(MODEL_DIR, Path(args.teacher), len(label_names), device)
    for p in teacher.parameters():
        p.requires_grad_(False)
    # eval() 关闭 dropout 等训练期行为，保证教师对同一输入给出稳定输出
    teacher.eval()
    student = BiLSTMClassifier(vocab_size, len(label_names),
                               embed_size=args.embed_size,
                               hidden_size=args.hidden_size).to(device)
    # 教师参数量不能用 count_parameters 统计：它只累加 requires_grad=True 的参数，
    # 而教师在上面刚被整体冻结，会得出 0.0M 的错误结果。
    # 学生仍在训练，requires_grad 全为 True，可以正常用 count_parameters。
    teacher_params = sum(p.numel() for p in teacher.parameters())
    logger.info(f"教师参数量={teacher_params / 1e6:.1f}M "
                f"学生参数量={count_parameters(student) / 1e6:.1f}M"
                f"（{count_parameters(student) / teacher_params:.1%}）")

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
    loss_per_epoch = []
    for ep in range(args.epochs):
        student.train()
        t0 = time.time()
        total_loss = 0.0
        for batch in train_dl:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            true_labels = batch["label"].to(device)
            opt.zero_grad()
            # 教师已冻结：用 no_grad 关掉梯度记录，省显存也省时间
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
            # 梯度裁剪：LSTM 在长序列上容易梯度爆炸，把梯度范数限制在 1.0 以内
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()

        epoch_loss = total_loss / len(train_dl)
        loss_per_epoch.append(epoch_loss)
        dev_acc = evaluate_bilstm(student, dev_dl, device)
        logger.info(f"epoch {ep + 1}/{args.epochs} loss={epoch_loss:.4f} "
                    f"dev_acc={dev_acc:.4f} 用时={time.time() - t0:.0f}s")
        # 用 dev 选最优权重；dev 只有 8 类，选择信号不完整，但不会造成
        # 测试集信息泄漏——最终指标仍在 test 上单独计算
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            torch.save(student.state_dict(), out_dir / "best.pt")

    # ---------- 在测试集上报告教师与学生的完整指标 ----------
    # 三件套的延迟必须在同一设备上比较（量化后的 INT8 算子只有 CPU 实现），
    # 因此这里把两个模型都搬到 CPU 后再测体积与延迟
    test_texts, test_labels = load_any(Path(args.input), label_names)
    logger.info(f"评估数据: {args.input}（{len(test_texts)} 条，{len(label_names)} 类）")
    teacher.to("cpu")
    student.to("cpu")

    teacher_metrics = metrics_from_predictions(
        predict_all(teacher, tokenizer, test_texts, "cpu"), test_labels,
        label_names)
    test_dl = build_dataloader(test_texts, test_labels, tokenizer, args.bs, MAX_LEN)
    # predict_bilstm_all 返回 (预测列表, 真实标签列表) 元组，这里只取预测列表；
    # 真实标签用 test_labels，避免两个来源混用导致长度不一致
    student_predictions, _ = predict_bilstm_all(student, test_dl, "cpu")
    student_metrics = metrics_from_predictions(
        student_predictions, test_labels, label_names)

    teacher_metrics["size_mb"] = measure_size_mb(teacher)
    student_metrics["size_mb"] = measure_size_mb(student)
    teacher_metrics["latency_ms"] = bench_latency_bert(teacher, tokenizer, "cpu")
    student_metrics["latency_ms"] = bench_latency_bilstm(student, tokenizer, "cpu")

    logger.info(f"教师: acc={teacher_metrics['accuracy']:.4f} "
                f"macro_f1={teacher_metrics['macro_f1']:.4f} "
                f"体积={teacher_metrics['size_mb']:.1f}MB "
                f"延迟={teacher_metrics['latency_ms']:.1f}ms")
    logger.info(f"学生: acc={student_metrics['accuracy']:.4f} "
                f"macro_f1={student_metrics['macro_f1']:.4f} "
                f"体积={student_metrics['size_mb']:.1f}MB "
                f"延迟={student_metrics['latency_ms']:.1f}ms")

    student_ckpt = out_dir / "best.pt"
    predicted = verify_student_artifact(student_ckpt, vocab_size, len(label_names),
                                       tokenizer, test_texts[0])
    logger.info(f"产物验证: 重新加载并推理首条样本 -> 标签 ID {predicted}"
                f"（真实 {test_labels[0]}）")

    (out_dir / "metrics.json").write_text(json.dumps({
        "args": vars(args),
        "best_dev_acc": best_dev_acc,
        "student_params": count_parameters(student),
        "student_vocab_size": vocab_size,
        "loss_per_epoch": loss_per_epoch,
        "sample_count": len(test_texts),
        "teacher": teacher_metrics,
        "student": student_metrics,
        "macro_f1_delta": student_metrics["macro_f1"] - teacher_metrics["macro_f1"],
        "size_ratio": student_metrics["size_mb"] / teacher_metrics["size_mb"],
        "artifact_bytes": student_ckpt.stat().st_size,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"蒸馏完成 best_dev_acc={best_dev_acc:.4f}，"
                f"权重已保存到 {student_ckpt}，指标已写入 {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="BERT -> BiLSTM 软标签知识蒸馏")
    ap.add_argument("--teacher", default=str(CKPT_DIR / "intent_bert" / "best.pt"),
                    help="教师模型权重路径")
    ap.add_argument("--input", default=str(TEST_FILE),
                    help="最终评估数据文件，默认 data/test.txt（完整 9 类）")
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
