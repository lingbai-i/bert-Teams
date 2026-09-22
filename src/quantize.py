"""动态量化：将微调后 BERT 的 Linear 层权重量化为 INT8。

动态量化（DQ）是训练后量化中最简单的方式：无需校准数据、无需重新训练，
推理时权重从 FP32 实时转 INT8 计算再转回。仅支持 CPU 推理。

实测结论（本项目模板合成数据，教师为 202500 条全量训练、seed=42）：
    体积：390.2 MB -> 145.6 MB，为原来的 37.3%，与预期一致。
    精度：准确率 0.9844 -> 0.9842，宏平均 F1 0.9846 -> 0.9844，差异在千分之一量级。
    延迟：CPU 单条延迟没有改善，反而略慢（66.9ms -> 80.9ms）。
        原因：动态量化只压 Linear 层，而 max_len=48 的短序列下 BERT 的 CPU 耗时
        大量落在注意力矩阵乘、LayerNorm、GELU 这些未被量化的算子上；
        同时每次前向还要为量化层做一次「INT8 -> 浮点」的反量化。
        因此本模块只承诺省体积，不承诺提速。

评估口径：默认在 data/test.txt（完整 9 类）上评估，可用 --input 覆盖，
也支持 data/eval/test_queries.jsonl 格式的人工评估集。
刻意不用 data/dev.txt：它只有 8 个类别（缺 other_chitchat），算出的指标与测试集
口径不可比；且 sklearn 的 classification_report 在类别数与标签名数量不一致时会
直接抛 ValueError（团队此前的 evaluate.py 在 dev 上就是这样崩的）。

体积口径：取 state_dict 序列化到临时文件后的真实字节数，不用
src.model.state_dict_size_mb。后者逐张量求和，而动态量化把 Linear 权重打包进
packed params（不是 Tensor），会被整块漏掉。同一模型两种算法的实测差距：
    真实序列化 145.6 MB  vs  逐张量求和 63.5 MB
即真实压缩比 37.3%，逐张量求和会得出 16.3% 的假结论。

用法：
    python -m src.quantize --ckpt checkpoints/intent_bert/best.pt
"""
from pathlib import Path
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


def verify_quantized_artifact(teacher_ckpt: Path, quantized_ckpt: Path,
                              num_labels: int, tokenizer, device: str,
                              sample_text: str) -> int:
    """验证保存的 INT8 权重能否被独立加载并推理。

    参数:
        teacher_ckpt: 教师 FP32 权重路径，用于构建量化前的模型结构。
        quantized_ckpt: 量化权重文件路径。
        num_labels: 类别数。
        tokenizer: 分词器。
        device: 推理设备，固定为 CPU。
        sample_text: 用于验证推理链路的文本。

    返回:
        预测出的标签 ID。

    异常:
        FileNotFoundError: 量化权重文件不存在时抛出。

    说明（加载顺序约束）：
        必须按「用 FP32 权重建模型 -> 量化 -> 加载 INT8 状态字典」的顺序加载。
        原因：量化把 nn.Linear 替换成了量化模块，模块类型已经改变，
        因此把 INT8 状态字典直接加载到未量化的模型上会报键名不匹配。
        部署层 src/predict.py 的 BertPredictor 也是按这个顺序加载量化模型的。
    """
    if not quantized_ckpt.exists():
        raise FileNotFoundError(f"未找到量化权重：{quantized_ckpt}")
    probe = load_classifier(MODEL_DIR, teacher_ckpt, num_labels, device)
    probe = torch.quantization.quantize_dynamic(
        probe, {torch.nn.Linear}, dtype=torch.qint8)
    state = torch.load(str(quantized_ckpt), map_location=device,
                       weights_only=True)
    probe.load_state_dict(state)
    probe.eval()
    enc = tokenizer(sample_text, return_tensors="pt", truncation=True,
                    max_length=MAX_LEN)
    with torch.no_grad():
        predicted = int(probe(**enc).logits.argmax(-1)[0])
    # 显式释放这份临时模型，避免它一直占着内存
    del probe
    return predicted


def main(args) -> None:
    """执行量化、评估、保存与产物验证的完整流程。

    参数:
        args: 命令行参数，含 ckpt、input、output_dir。

    返回:
        None。

    异常:
        FileNotFoundError: 教师权重不存在时由 load_classifier 抛出。
    """
    # 精度评估用 GPU（若可用）以加速：实测 11099 条在 GPU 上约 3.5 秒、
    # 在 CPU 上需 188 秒，而两者算出的准确率完全一致（都是 0.9840）。
    eval_device = "cuda" if torch.cuda.is_available() else "cpu"
    # 量化与延迟测量必须在 CPU：INT8 算子没有 CUDA 实现，放 GPU 会在前向时直接报错
    latency_device = "cpu"
    label_names = load_labels(CLASS_FILE)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = load_classifier(MODEL_DIR, Path(args.ckpt), len(label_names),
                            eval_device)

    texts, labels = load_any(Path(args.input), label_names)
    logger.info(f"评估数据: {args.input}（{len(texts)} 条，{len(label_names)} 类）")

    metrics_before = compute_metrics(model, tokenizer, texts, labels,
                                     label_names, eval_device)
    size_before = measure_size_mb(model)
    # 延迟测量前把模型搬到 CPU；量化本身也要求模型在 CPU 上
    model.to(latency_device)
    latency_before = bench_latency_bert(model, tokenizer, latency_device)
    logger.info(f"量化前: acc={metrics_before['accuracy']:.4f} "
                f"macro_f1={metrics_before['macro_f1']:.4f} "
                f"体积={size_before:.1f}MB CPU延迟={latency_before:.1f}ms")

    quantized = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8)
    # 量化后的 INT8 模型只能跑在 CPU 上，评估与延迟测量都在 CPU 做
    metrics_after = compute_metrics(quantized, tokenizer, texts, labels,
                                    label_names, latency_device)
    size_after = measure_size_mb(quantized)
    latency_after = bench_latency_bert(quantized, tokenizer, latency_device)
    logger.info(f"量化后: acc={metrics_after['accuracy']:.4f} "
                f"macro_f1={metrics_after['macro_f1']:.4f} "
                f"体积={size_after:.1f}MB CPU延迟={latency_after:.1f}ms")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "best_int8.pt"
    torch.save(quantized.state_dict(), ckpt_path)
    artifact_bytes = ckpt_path.stat().st_size
    logger.info(f"量化模型已保存到 {ckpt_path}"
                f"（{artifact_bytes / 1024 / 1024:.1f}MB）")

    predicted = verify_quantized_artifact(
        Path(args.ckpt), ckpt_path, len(label_names), tokenizer,
        latency_device, texts[0])
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
        "size_ratio": size_after / size_before,
        "cpu_latency_ms_before": latency_before,
        "cpu_latency_ms_after": latency_after,
        "artifact_bytes": artifact_bytes,
        "report_after": metrics_after["report"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"指标已写入 {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="BERT 动态量化（INT8，仅 CPU）")
    ap.add_argument("--ckpt", default=str(CKPT_DIR / "intent_bert" / "best.pt"),
                    help="教师权重路径")
    ap.add_argument("--input", default=str(TEST_FILE),
                    help="评估数据文件，默认 data/test.txt（完整 9 类）")
    ap.add_argument("--output-dir", default=str(CKPT_DIR / "intent_bert_int8"))
    main(ap.parse_args())
