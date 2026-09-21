"""压缩三件套共享的度量工具。

职责：为动态量化、非结构化剪枝、知识蒸馏三个压缩模块提供同一套度量实现，
消除三份重复代码，并保证「精度-体积-延迟」对比表的口径一致。

包含三组接口：
1. 体积：measure_size_mb —— 取 state_dict 序列化后的真实字节数
2. 延迟：bench_latency_bert / bench_latency_bilstm —— 单条文本的纯前向耗时
3. 精度：compute_metrics（BERT 系列）、metrics_from_predictions（由预测算指标）

为什么体积要取真实序列化大小：
    不用 src.model.state_dict_size_mb。该函数逐张量求和，而动态量化把 Linear
    权重打包进 packed params（不是 Tensor），会被整块漏掉。实测同一量化模型：
        真实序列化大小 145.6 MB，逐张量求和只有 63.5 MB
    即真实压缩比 37.3%，用后者会得出 16.3% 的假结论。
    另外：剪枝后调用本函数会得到与剪枝前相同的数值，这不是 bug——
    非结构化剪枝不改变张量形状，元素个数与字节数都不变。

延迟口径：
    计时对象是纯前向传播：分词与设备搬运在计时循环之外只做一次。
    因此该数字不含分词与 argmax 的开销，比真实请求的端到端耗时略低；
    三件套使用同一实现，口径一致，可横向比较。
    延迟受机器负载影响极大（实测同一模型波动可达 3.4 倍），
    因此只有同一次会话内相邻测出的相对关系才可信。

设备策略：
    精度评估用 GPU（若可用）以加速——实测 11099 条在 GPU 上约 3.5 秒，
    在 CPU 上需 188 秒，而两者算出的准确率完全一致（都是 0.9840）。
    延迟测量必须在 CPU 上做：量化后的 INT8 算子只有 CPU 实现，
    三件套必须在同一设备口径下比较才有意义。

依赖关系：
- 上游：src.config 提供 MAX_LEN；src.evaluate 提供 predict_all
- 下游：src/quantize.py、src/prune.py、src/distill.py
- 第三方：torch、scikit-learn

用法：
    由三个压缩模块导入调用，不单独执行。
"""
from pathlib import Path
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score)
from torch.utils.data import DataLoader
from typing import Any, Dict, List, Tuple

import os
import tempfile
import time

import torch

from src.config import MAX_LEN
from src.evaluate import predict_all

# 延迟测量用的探针问句。三件套统一使用同一句，保证输入长度一致、数字可比
LATENCY_PROBE_TEXT = "今天下午的会议改到几点了"


def measure_size_mb(model) -> float:
    """测量模型序列化后的真实体积。

    参数:
        model: 待测量模型。

    返回:
        体积，单位 MB。

    说明:
        做法是把 state_dict 写进系统临时文件、取字节数、再删除，
        因此不污染任何输出目录。

        为什么不用 src.model.state_dict_size_mb：那个函数遍历 state_dict
        逐张量求和，而动态量化后的 Linear 权重存放在 packed params 中
        （不是 Tensor），会被整块漏掉。实测同一量化模型：
            真实序列化大小 145.6 MB，逐张量求和只有 63.5 MB
        即真实压缩比 37.3%，用后者会得出 16.3% 的假结论。

        剪枝后调用本函数会得到与剪枝前相同的数值——这不是 bug，
        而是非结构化剪枝的固有性质：张量形状未变，元素个数与字节数都不变。

        副作用：会经历一次完整的磁盘序列化，BERT 规模上耗时数秒。
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


def bench_latency_bert(model, tokenizer, device: str = "cpu",
                       n: int = 100) -> float:
    """测量 BERT 系列模型单条文本的平均推理延迟。

    参数:
        model: BERT 系列模型。
        tokenizer: 分词器。
        device: 推理设备。三件套统一传 "cpu" 以保证可比性。
        n: 测量次数，取均值，默认 100。

    返回:
        平均延迟，单位毫秒。

    说明:
        计时对象是纯前向传播：分词与设备搬运在计时循环之外只做一次，
        循环内只调用 model(**enc)。因此该数字不含分词与 argmax 的开销，
        比真实请求的端到端耗时略低。三件套使用同一实现，口径一致，可横向比较。

        测量前先执行 10 次预热，排除首次调用时算子选择与内存分配的开销。
        延迟受机器负载影响极大（实测同模型波动可达 3.4 倍），
        只有同一次会话内相邻测出的相对关系才可信。
    """
    enc = tokenizer(LATENCY_PROBE_TEXT, return_tensors="pt",
                    truncation=True, max_length=MAX_LEN)
    enc = {key: value.to(device) for key, value in enc.items()}
    model.eval()
    with torch.no_grad():
        for _ in range(10):  # 预热，排除首次调用的初始化开销
            model(**enc)
        t0 = time.perf_counter()
        for _ in range(n):
            model(**enc)
    return (time.perf_counter() - t0) / n * 1000


def bench_latency_bilstm(model, tokenizer, device: str = "cpu",
                         n: int = 100) -> float:
    """测量 BiLSTM 学生模型单条文本的平均推理延迟。

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
        计时对象同样是纯前向传播，不含分词与 argmax。
    """
    enc = tokenizer(LATENCY_PROBE_TEXT, return_tensors="pt",
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


def predict_bilstm_all(model, dl: DataLoader,
                       device: str) -> Tuple[List[int], List[int]]:
    """遍历 DataLoader 收集 BiLSTM 学生模型的预测与真实标签。

    参数:
        model: 学生模型。
        dl: 数据加载器。
        device: 推理设备。

    返回:
        (预测标签 ID 列表, 真实标签 ID 列表)，两者等长且同序。

    说明:
        学生不能复用 src.evaluate.predict_all：那个函数按
        model(input_ids=..., attention_mask=...).logits 调用，
        而 BiLSTMClassifier 的 forward 直接返回 logits 张量、没有 logits 属性。
        因此单独实现一份，只取返回值的张量本身。
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
        显式传 labels 后，缺席类别会以 support=0 的形式出现在报告里，
        是可见的，而不是被静默忽略。

        宏平均与准确率的区别：准确率受各类样本数影响；宏平均先按类别各算一个值
        再对类别等权平均，少数类的表现不会被多数类淹没。两者差距大说明模型
        在某些类别上明显偏弱。
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


def compute_metrics(model, tokenizer, texts: List[str], gold_labels: List[int],
                    label_names: List[str], device: str) -> Dict[str, Any]:
    """计算 BERT 系列模型的准确率与宏平均 precision/recall/F1。

    参数:
        model: 待评估的 BERT 系列模型。
        tokenizer: 分词器。
        texts: 文本列表。
        gold_labels: 真实标签 ID 列表，与 texts 等长。
        label_names: 标签名列表，下标即标签 ID。
        device: 推理设备。建议传 GPU 以加速（准确率与设备无关）。

    返回:
        同 metrics_from_predictions 的返回结构。

    说明:
        批量前向复用 src.evaluate.predict_all，本模块不重复实现推理循环——
        该函数按 model(input_ids=..., attention_mask=...).logits 调用，
        与 BERT 系列的输出契约一致。
    """
    predictions = predict_all(model, tokenizer, texts, device)
    return metrics_from_predictions(predictions, gold_labels, label_names)
