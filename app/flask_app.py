"""Flask 部署入口：多模型意图分类 Web 服务。

接口：
    GET  /             聊天式测试页面（可多模型对比）
    GET  /api/models   当前可用的模型列表
    POST /api/predict  意图预测，请求体 {"text": "...", "model": "bert"}

启动：
    python -m app.flask_app --port 5000
"""
from flask import Flask, jsonify, render_template, request
from typing import Dict, List

import argparse
import logging
import time

from src.predict import get_available_models, get_predictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
DEVICE = "cpu"


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/models")
def models():
    return jsonify({"ok": True, "models": get_available_models()})


@app.post("/api/predict")
def predict():
    """执行单条文本的意图预测。

    请求体: {"text": 待分类文本, "model": 模型名，缺省为 bert}

    返回:
        成功时为预测结果 JSON（含 intent / confidence / latency_ms / top3）。
        失败时统一返回 {"ok": false, "error": 原因} 形式的 JSON，
        使前端始终能解析出可读原因，而不是收到 HTML 错误页。
    """
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    model_name = data.get("model") or "bert"
    if not text:
        return jsonify({"ok": False, "error": "text 不能为空"}), 400
    try:
        result = get_predictor(model_name, DEVICE).predict(text)
    except FileNotFoundError as e:
        # 训练产物不存在：调用方需先完成对应路线的训练
        return jsonify({"ok": False, "error": str(e)}), 400
    except ValueError as e:
        # 模型名未注册：属于调用方参数错误
        return jsonify({"ok": False, "error": str(e)}), 400
    except (RuntimeError, ImportError) as e:
        # 模型已注册但当前环境跑不起来（如 LLM 路线缺 DEEPSEEK_API_KEY、
        # 未安装 fasttext）：返回 503，避免前端只拿到 500 空白页
        return jsonify({"ok": False,
                        "error": f"模型 {model_name} 当前不可用：{e}"}), 503
    result.update({"ok": True, "model": model_name})
    return jsonify(result)


def warm_up(models: List[Dict[str, str]]) -> None:
    """启动时预加载全部可用模型，把权重加载开销前移到服务启动阶段。

    现象：首次调用 /api/predict 时界面长时间无响应（CPU 上加载 BERT 约需数十秒），
          而接口返回的 latency_ms 仅数十毫秒，两者相差数百倍。
    原因：get_predictor 采用进程内惰性缓存，权重直到首个请求所在的线程才加载；
          而 src/predict.py 的 latency_ms 只统计 predict() 内部耗时，不含加载。
    后果：演示时首次点击「开始分类」出现无响应空窗，且响应时间指标失真。
          预热后加载耗时只体现在启动日志，请求侧保留纯推理耗时。

    参数:
        models: get_available_models() 返回的模型列表，每项含 name 与 desc 字段。

    异常:
        任一模型加载失败即向上抛出，让服务在启动阶段就暴露问题，
        避免带着不可用模型启动，直到演示时才在请求侧失败。
    """
    for item in models:
        name = item["name"]
        t0 = time.perf_counter()
        get_predictor(name, DEVICE)
        logger.info(f"模型 {name} 预热完成，耗时 {time.perf_counter() - t0:.1f} 秒")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="意图分类 Flask 服务")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--device", default="cpu",
                    help="BERT 系列推理设备，部署默认 CPU")
    args = ap.parse_args()
    DEVICE = args.device
    available = get_available_models()
    logger.info(f"可用模型: {[m['name'] for m in available]}")
    warm_up(available)
    app.run(host=args.host, port=args.port)
