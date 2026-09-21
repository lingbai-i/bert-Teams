"""统一推理封装：为部署层提供多模型一致的 predict 接口。

所有预测器返回统一结构：
    {"intent": 标签名, "confidence": 置信度或 None, "latency_ms": 毫秒,
     "top3": [(标签名, 概率), ...]（仅 BERT 系列提供）}

支持的模型类型（按产物是否存在自动探测）：
    bert      微调后的 BERT（含剪枝版，结构相同）
    bert_int8 动态量化版 BERT（仅 CPU）
    bilstm    蒸馏得到的 BiLSTM 学生模型（仅 CPU 部署场景）
    rf        TF-IDF + 随机森林
    fasttext  fasttext 词级别分类
    llm       远程大模型 API（需配置 DEEPSEEK_API_KEY）
"""
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv

import os
import pickle
import time

import jieba
import torch
import torch.nn.functional as F

from src.config import CKPT_DIR, CLASS_FILE, MAX_LEN, MODEL_DIR
from src.data import load_labels

# 各模型产物的默认路径与说明，get_available_models 据此探测
MODEL_REGISTRY = {
    "bert": ("BERT 微调模型", CKPT_DIR / "intent_bert" / "best.pt"),
    "bert_pruned": ("BERT 剪枝模型", CKPT_DIR / "intent_bert_pruned" / "best.pt"),
    "bert_int8": ("BERT 动态量化模型", CKPT_DIR / "intent_bert_int8" / "best_int8.pt"),
    "bilstm": ("BiLSTM 蒸馏模型", CKPT_DIR / "intent_bilstm_distill" / "best.pt"),
    "rf": ("随机森林基线", CKPT_DIR / "rf" / "rf.pkl"),
    "fasttext": ("fasttext 基线", CKPT_DIR / "fasttext" / "ft.bin"),
    "llm": ("远程大模型", None),
}


class BasePredictor:
    """预测器基类，约定 predict 返回结构。"""

    def predict(self, text: str) -> dict:
        """对单条文本预测意图。

        参数:
            text: 用户输入文本。

        返回:
            统一结构的预测结果字典，latency_ms 为本次推理耗时（毫秒）。
        """
        raise NotImplementedError


class BertPredictor(BasePredictor):
    """BERT 系列（微调 / 剪枝 / 动态量化）预测器，三者在 CPU 上结构兼容。"""

    def __init__(self, ckpt: Path, device: str = "cpu", quantized: bool = False):
        from transformers import AutoTokenizer

        from src.model import build_classifier, count_parameters

        self.device = "cpu" if quantized else device
        self.labels = load_labels(CLASS_FILE)
        self.tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
        model = build_classifier(MODEL_DIR, len(self.labels), self.device)
        if quantized:
            # 先量化再加载 INT8 state_dict，保证模块类型与权重格式一致
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8)
        state = torch.load(str(ckpt), map_location=self.device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        self.model = model

    def predict(self, text: str) -> dict:
        t0 = time.perf_counter()
        enc = self.tokenizer(text, return_tensors="pt", truncation=True,
                             max_length=MAX_LEN)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            probs = F.softmax(self.model(**enc).logits, -1).cpu().numpy()[0]
        top3 = probs.argsort()[::-1][:3]
        return {
            "intent": self.labels[top3[0]],
            "confidence": float(probs[top3[0]]),
            "latency_ms": (time.perf_counter() - t0) * 1000,
            "top3": [(self.labels[i], float(probs[i])) for i in top3],
        }


class BiLSTMPredictor(BasePredictor):
    """BiLSTM 蒸馏学生模型预测器（结构与训练时保持一致）。"""

    def __init__(self, ckpt: Path, device: str = "cpu"):
        from transformers import AutoTokenizer

        from src.model import BiLSTMClassifier

        self.device = device
        self.labels = load_labels(CLASS_FILE)
        self.tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
        model = BiLSTMClassifier(self.tokenizer.vocab_size, len(self.labels))
        state = torch.load(str(ckpt), map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.to(device).eval()
        self.model = model

    def predict(self, text: str) -> dict:
        t0 = time.perf_counter()
        enc = self.tokenizer(text, return_tensors="pt", truncation=True,
                             max_length=MAX_LEN)
        with torch.no_grad():
            probs = F.softmax(
                self.model(enc["input_ids"].to(self.device),
                           enc["attention_mask"].to(self.device)), -1
            ).cpu().numpy()[0]
        top3 = probs.argsort()[::-1][:3]
        return {
            "intent": self.labels[top3[0]],
            "confidence": float(probs[top3[0]]),
            "latency_ms": (time.perf_counter() - t0) * 1000,
            "top3": [(self.labels[i], float(probs[i])) for i in top3],
        }


class RFPredictor(BasePredictor):
    """随机森林预测器，依赖同目录下的 tfidf.pkl 与 rf.pkl。"""

    def __init__(self, model_dir: Path):
        from src.rf_baseline import load_stopwords

        self.labels = load_labels(CLASS_FILE)
        self.stopwords = load_stopwords()
        with open(model_dir / "tfidf.pkl", "rb") as f:
            self.tfidf = pickle.load(f)
        with open(model_dir / "rf.pkl", "rb") as f:
            self.model = pickle.load(f)

    def predict(self, text: str) -> dict:
        t0 = time.perf_counter()
        words = " ".join(w for w in jieba.lcut(text)
                         if w.strip() and w not in self.stopwords)
        pred = int(self.model.predict(self.tfidf.transform([words]))[0])
        return {
            "intent": self.labels[pred],
            "confidence": None,
            "latency_ms": (time.perf_counter() - t0) * 1000,
        }


class FastTextPredictor(BasePredictor):
    """fasttext 预测器（需在安装了 fasttext 的环境中运行）。"""

    def __init__(self, bin_path: Path):
        import fasttext

        self.model = fasttext.load_model(str(bin_path))

    def predict(self, text: str) -> dict:
        t0 = time.perf_counter()
        words = " ".join(w for w in jieba.lcut(text) if w.strip())
        labels, probs = self.model.predict(words, k=1)
        return {
            "intent": labels[0].replace("__label__", ""),
            "confidence": float(probs[0]),
            "latency_ms": (time.perf_counter() - t0) * 1000,
        }


class LLMPredictor(BasePredictor):
    """远程大模型预测器，每次调用产生 API 请求，延迟约 1~2 秒。"""

    def __init__(self):
        load_dotenv()
        self.api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise RuntimeError("未配置 DEEPSEEK_API_KEY，无法使用 LLM 路线")

    def predict(self, text: str) -> dict:
        from src.llm_baseline import classify

        t0 = time.perf_counter()
        intent = classify(text, self.api_key)
        return {
            "intent": intent or "解析失败",
            "confidence": None,
            "latency_ms": (time.perf_counter() - t0) * 1000,
        }


def get_available_models() -> List[Dict[str, str]]:
    """探测当前可用的模型列表（按产物文件是否存在判断）。"""
    available = []
    for name, (desc, artifact) in MODEL_REGISTRY.items():
        if name == "llm":
            load_dotenv()
            ok = bool(os.environ.get("DEEPSEEK_API_KEY"))
        else:
            ok = artifact is not None and artifact.exists()
        if ok:
            available.append({"name": name, "desc": desc})
    return available


_predictor_cache: Dict[str, BasePredictor] = {}


def get_predictor(name: str, device: str = "cpu") -> BasePredictor:
    """按名称获取预测器（进程内缓存，避免重复加载模型）。

    参数:
        name: MODEL_REGISTRY 中的模型名。
        device: BERT 系列的推理设备，量化与树模型强制 CPU。

    异常:
        ValueError: 模型名未注册。
        FileNotFoundError: 对应训练产物不存在，提示先训练。
    """
    if name in _predictor_cache:
        return _predictor_cache[name]
    if name not in MODEL_REGISTRY:
        raise ValueError(f"未注册的模型：{name}，可选：{list(MODEL_REGISTRY)}")
    artifact = MODEL_REGISTRY[name][1]
    if name == "bert":
        predictor = BertPredictor(artifact, device)
    elif name == "bert_pruned":
        predictor = BertPredictor(artifact, device)
    elif name == "bert_int8":
        predictor = BertPredictor(artifact, quantized=True)
    elif name == "bilstm":
        predictor = BiLSTMPredictor(artifact, device)
    elif name == "rf":
        predictor = RFPredictor(artifact.parent)
    elif name == "fasttext":
        predictor = FastTextPredictor(artifact)
    else:
        predictor = LLMPredictor()
    _predictor_cache[name] = predictor
    return predictor
