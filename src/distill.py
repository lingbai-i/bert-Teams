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
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import argparse
import json
import logging
import time

import torch
import torch.nn.functional as F

from src.compress_bench import (bench_latency_bert, bench_latency_bilstm,
                                compute_metrics, measure_size_mb,
                                metrics_from_predictions, predict_bilstm_all)
from src.config import (
    BATCH_SIZE, CKPT_DIR, CLASS_FILE, DEV_FILE, MAX_LEN, MODEL_DIR, SEED,
    TEST_FILE, TRAIN_FILE,
)
from src.data import build_dataloader, load_labels, load_txt, subsample
from src.evaluate import load_any
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
    # 精度评估用 GPU 加速（若可用）：实测 11099 条 GPU 约 3.5 秒、CPU 需 188 秒，
    # 而准确率与设备无关（两种设备算出的教师准确率都是 0.9840）。
    # 延迟测量必须在 CPU 上做——量化后的 INT8 算子只有 CPU 实现，
    # 三件套必须在同一设备口径下比较才有意义。
    test_texts, test_labels = load_any(Path(args.input), label_names)
    logger.info(f"评估数据: {args.input}（{len(test_texts)} 条，{len(label_names)} 类）")
    eval_device = "cuda" if torch.cuda.is_available() else "cpu"
    latency_device = "cpu"

    # 评估前必须重新加载 dev 最优权重。
    # 原因：训练循环结束后 student 内存里是「最后一轮」的权重，而交付产物是 dev
    # 准确率最高那一轮保存的 best.pt，两者可能不是同一轮——本次实测 best 出现在
    # 第 2 轮（dev_acc 0.9831），第 3 轮反而降到 0.9830。若直接用内存中的 student
    # 出指标，metrics.json 报的就不是交付产物 best.pt 的精度。
    student_ckpt = out_dir / "best.pt"
    if not student_ckpt.exists():
        raise FileNotFoundError(
            f"未找到最优权重 {student_ckpt}：训练过程中 dev 准确率从未提升，"
            f"无法确定交付产物，请检查验证集是否为空")
    student.load_state_dict(
        torch.load(str(student_ckpt), map_location="cpu", weights_only=True))
    logger.info(f"已重新加载 dev 最优权重（best_dev_acc={best_dev_acc:.4f}）"
                f"用于最终评估")

    # 教师是 BERT 系列，可直接用 compress_bench.compute_metrics 走统一路径
    teacher_metrics = compute_metrics(teacher, tokenizer, test_texts, test_labels,
                                      label_names, eval_device)
    test_dl = build_dataloader(test_texts, test_labels, tokenizer, args.bs, MAX_LEN)
    # predict_bilstm_all 返回 (预测列表, 真实标签列表) 元组，这里只取预测列表；
    # 真实标签用 test_labels，避免两个来源混用导致长度不一致
    student_predictions, _ = predict_bilstm_all(student, test_dl, eval_device)
    student_metrics = metrics_from_predictions(
        student_predictions, test_labels, label_names)

    teacher_metrics["size_mb"] = measure_size_mb(teacher)
    student_metrics["size_mb"] = measure_size_mb(student)
    # 延迟测量搬到 CPU 上进行，与量化/剪枝保持同一口径
    teacher.to(latency_device)
    student.to(latency_device)
    teacher_metrics["latency_ms"] = bench_latency_bert(teacher, tokenizer,
                                                       latency_device)
    student_metrics["latency_ms"] = bench_latency_bilstm(student, tokenizer,
                                                         latency_device)

    logger.info(f"教师: acc={teacher_metrics['accuracy']:.4f} "
                f"macro_f1={teacher_metrics['macro_f1']:.4f} "
                f"体积={teacher_metrics['size_mb']:.1f}MB "
                f"延迟={teacher_metrics['latency_ms']:.1f}ms")
    logger.info(f"学生: acc={student_metrics['accuracy']:.4f} "
                f"macro_f1={student_metrics['macro_f1']:.4f} "
                f"体积={student_metrics['size_mb']:.1f}MB "
                f"延迟={student_metrics['latency_ms']:.1f}ms")

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
