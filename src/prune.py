"""全局非结构化剪枝：按比例置零 BERT 各层注意力 query 权重中绝对值最小的参数。

非结构化剪枝属于「逻辑剪枝」：权重被置零但张量形状不变，因此模型体积不变、
CPU 推理提速有限；其价值在于验证网络中存在大量冗余参数（精度几乎不掉），
配合稀疏格式存储或专用推理库才能进一步带来体积/速度收益。

实测结论（本项目模板合成数据，教师为 202500 条全量训练、seed=42）：
    精度：准确率 0.9844、宏平均 F1 0.9846，剪枝前后完全一致，badcase 数量也未变。
    体积：390.2 MB -> 390.2 MB，一字未变，这是非结构化剪枝的固有性质。
    延迟：68.8ms -> 71.5ms，差异在测量噪声范围内，没有实质改善。
    稀疏度：0% -> 30.00%。
    即「注意力 query 权重中 30% 可以直接置零而不损失任何精度」，
    这是本模块在对比表里的真正产出，不能用体积或延迟冒充效果。

评估口径：默认在 data/test.txt（完整 9 类）上评估，可用 --input 覆盖。
刻意不用 data/dev.txt：它只有 8 个类别（缺 other_chitchat），算出的指标与测试集
口径不可比；且其标签数与分类报告的 target_names 不一致时 sklearn 会直接报错。

延迟口径：推理评估用 GPU 以加快速度，但延迟统一搬到 CPU 上测——
量化后的 INT8 算子只有 CPU 实现，三件套必须在同一设备口径下比较。

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

from src.compress_bench import (bench_latency_bert, compute_metrics,
                                measure_size_mb)
from src.config import CLASS_FILE, CKPT_DIR, MAX_LEN, MODEL_DIR, TEST_FILE
from src.data import load_labels
from src.evaluate import load_any
from src.model import load_classifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def query_sparsity(model) -> float:
    """计算各 encoder 层注意力 query 权重的整体稀疏度（零值占比）。

    参数:
        model: BERT 序列分类模型。

    返回:
        零值元素占比，取值 [0, 1]。例如 0.3 表示 30% 的权重为 0。

    异常:
        ZeroDivisionError: 模型不含任何 query 权重时抛出，属结构性错误。

    说明:
        稀疏度是剪枝唯一真正改变的指标，必须与体积、延迟一起报出来，
        否则「到底剪掉了什么」无从判断。
    """
    total = zeros = 0
    for layer in model.bert.encoder.layer:
        weight = layer.attention.self.query.weight
        total += weight.numel()
        zeros += (weight == 0).sum().item()
    if total == 0:
        raise ZeroDivisionError("未找到任何 query 权重，模型结构异常")
    return zeros / total


def verify_pruned_artifact(teacher_ckpt: Path, pruned_ckpt: Path,
                           num_labels: int, tokenizer, device: str,
                           sample_text: str) -> int:
    """验证保存的剪枝权重能否被独立加载并推理。

    参数:
        teacher_ckpt: 教师权重路径，用于构建模型结构。
        pruned_ckpt: 剪枝权重文件路径。
        num_labels: 类别数。
        tokenizer: 分词器。
        device: 推理设备。
        sample_text: 用于验证推理链路的文本。

    返回:
        预测出的标签 ID。

    异常:
        FileNotFoundError: 剪枝权重文件不存在时抛出。

    说明:
        剪枝不改变模型结构（张量形状未变），因此可以用普通 BERT 结构直接加载，
        这一点与量化产物不同——量化改变了模块类型，加载前必须先量化。
        这里刻意用「教师权重建模型 -> 加载剪枝权重」的顺序，
        验证的是剪枝权重本身可独立加载，而不是依赖剪枝时的内存状态。
    """
    if not pruned_ckpt.exists():
        raise FileNotFoundError(f"未找到剪枝权重：{pruned_ckpt}")
    probe = load_classifier(MODEL_DIR, teacher_ckpt, num_labels, device)
    state = torch.load(str(pruned_ckpt), map_location=device,
                       weights_only=True)
    probe.load_state_dict(state)
    probe.eval()
    enc = tokenizer(sample_text, return_tensors="pt", truncation=True,
                    max_length=MAX_LEN)
    with torch.no_grad():
        predicted = int(probe(**enc).logits.argmax(-1)[0])
    del probe
    return predicted


def main(args) -> None:
    """执行剪枝、评估、保存与产物验证的完整流程。

    参数:
        args: 命令行参数，含 ckpt、input、amount、output_dir。

    返回:
        None。

    异常:
        FileNotFoundError: 教师权重不存在时由 load_classifier 抛出。
    """
    # 评估用 GPU 加速（实测 11099 条 GPU 约 3.5 秒、CPU 需 188 秒，准确率一致）；
    # 延迟测量必须在 CPU 上做，故下面测延迟前会单独把模型搬过去
    device = "cuda" if torch.cuda.is_available() else "cpu"
    latency_device = "cpu"
    label_names = load_labels(CLASS_FILE)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = load_classifier(MODEL_DIR, Path(args.ckpt), len(label_names), device)

    texts, labels = load_any(Path(args.input), label_names)
    logger.info(f"评估数据: {args.input}（{len(texts)} 条，{len(label_names)} 类）")

    metrics_before = compute_metrics(model, tokenizer, texts, labels,
                                     label_names, device)
    size_before = measure_size_mb(model)
    sparsity_before = query_sparsity(model)
    # 剪枝前的延迟必须在剪枝之前测：剪枝是对模型原地修改的，
    # 一旦执行就再也拿不到剪枝前的权重，无法补测
    model.to(latency_device)
    latency_before = bench_latency_bert(model, tokenizer, latency_device)
    if device != latency_device:
        model.to(device)
    logger.info(f"剪枝前: acc={metrics_before['accuracy']:.4f} "
                f"macro_f1={metrics_before['macro_f1']:.4f} "
                f"体积={size_before:.1f}MB 稀疏度={sparsity_before:.2%} "
                f"CPU延迟={latency_before:.1f}ms")

    # 对所有 encoder 层的 query 权重做全局 L1 非结构化剪枝
    params = [(layer.attention.self.query, "weight")
              for layer in model.bert.encoder.layer]
    prune.global_unstructured(params, pruning_method=prune.L1Unstructured,
                              amount=args.amount)
    # 固化剪枝：把掩码合并进权重，移除重参数化钩子。
    #
    # 现象：不 remove 直接保存，state_dict 会多出 weight_orig 与 weight_mask
    #       两组键，文件体积约翻倍，部署时按普通键名加载会报键不匹配。
    # 原因：global_unstructured 不直接改权重，而是走重参数化机制——
    #       原权重改名为 weight_orig，再挂上 weight_mask 与前向钩子，
    #       前向时实时计算 weight_orig * mask。
    # 后果：remove 会把掩码乘进权重、删除掩码与钩子，产物才是一份与原始结构
    #       同构的普通权重。擅自去掉这一段会让产物无法被部署层加载。
    for module, name in params:
        prune.remove(module, name)

    metrics_after = compute_metrics(model, tokenizer, texts, labels,
                                    label_names, device)
    size_after = measure_size_mb(model)
    sparsity_after = query_sparsity(model)
    logger.info(f"剪枝后: acc={metrics_after['accuracy']:.4f} "
                f"macro_f1={metrics_after['macro_f1']:.4f} "
                f"体积={size_after:.1f}MB 稀疏度={sparsity_after:.2%}")

    # 延迟统一在 CPU 上测：量化后的 INT8 算子只有 CPU 实现，
    # 三件套必须在同一设备口径下比较延迟才有意义
    model.to(latency_device)
    latency_after = bench_latency_bert(model, tokenizer, latency_device)
    latency_change = (latency_after / latency_before - 1) * 100
    logger.info(f"剪枝后 CPU 延迟={latency_after:.1f}ms（"
                f"较剪枝前 {latency_change:+.1f}%）")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "best.pt"
    torch.save(model.state_dict(), ckpt_path)
    artifact_bytes = ckpt_path.stat().st_size
    logger.info(f"剪枝模型已保存到 {ckpt_path}"
                f"（{artifact_bytes / 1024 / 1024:.1f}MB）")

    predicted = verify_pruned_artifact(Path(args.ckpt), ckpt_path,
                                      len(label_names), tokenizer, "cpu",
                                      texts[0])
    logger.info(f"产物验证: 重新加载并推理首条样本 -> 标签 ID {predicted}"
                f"（真实 {labels[0]}）")

    (out_dir / "metrics.json").write_text(json.dumps({
        "args": vars(args),
        "sample_count": metrics_before["sample_count"],
        "accuracy_before": metrics_before["accuracy"],
        "accuracy_after": metrics_after["accuracy"],
        "macro_precision_before": metrics_before["macro_precision"],
        "macro_precision_after": metrics_after["macro_precision"],
        "macro_recall_before": metrics_before["macro_recall"],
        "macro_recall_after": metrics_after["macro_recall"],
        "macro_f1_before": metrics_before["macro_f1"],
        "macro_f1_after": metrics_after["macro_f1"],
        "size_mb_before": size_before,
        "size_mb_after": size_after,
        # 显式记录「体积未变」这一事实，避免读者误以为剪枝压缩了模型
        "size_unchanged": abs(size_after - size_before) < 1e-6,
        "sparsity_before": sparsity_before,
        "sparsity_after": sparsity_after,
        "cpu_latency_ms_before": latency_before,
        "cpu_latency_ms_after": latency_after,
        "artifact_bytes": artifact_bytes,
        "report_after": metrics_after["report"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"指标已写入 {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="BERT 全局非结构化剪枝")
    ap.add_argument("--ckpt", default=str(CKPT_DIR / "intent_bert" / "best.pt"),
                    help="教师权重路径")
    ap.add_argument("--input", default=str(TEST_FILE),
                    help="评估数据文件，默认 data/test.txt（完整 9 类）")
    ap.add_argument("--amount", type=float, default=0.3,
                    help="全局置零比例，取值 (0, 1)")
    ap.add_argument("--output-dir", default=str(CKPT_DIR / "intent_bert_pruned"))
    main(ap.parse_args())
