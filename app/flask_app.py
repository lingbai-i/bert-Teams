"""Flask 部署入口：多模型意图分类 Web 服务。

接口：
    GET  /             聊天式测试页面（可多模型对比）
    GET  /api/models   当前可用的模型列表
    POST /api/predict  意图预测，请求体 {"text": "...", "model": "bert"}

启动：
    python -m app.flask_app --port 5000
"""
from flask import Flask, jsonify, render_template, request

import argparse
import logging

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
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    model_name = data.get("model") or "bert"
    if not text:
        return jsonify({"ok": False, "error": "text 不能为空"}), 400
    try:
        result = get_predictor(model_name, DEVICE).predict(text)
    except FileNotFoundError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    result.update({"ok": True, "model": model_name})
    return jsonify(result)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="意图分类 Flask 服务")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--device", default="cpu",
                    help="BERT 系列推理设备，部署默认 CPU")
    args = ap.parse_args()
    DEVICE = args.device
    logger.info(f"可用模型: {[m['name'] for m in get_available_models()]}")
    app.run(host=args.host, port=args.port)
