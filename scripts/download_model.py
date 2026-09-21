"""下载 bert-base-chinese 预训练模型到 models/ 目录。

国内网络建议先设置镜像环境变量再运行：
    Windows PowerShell:  $env:HF_ENDPOINT="https://hf-mirror.com"
    Windows cmd:         set HF_ENDPOINT=https://hf-mirror.com
    Linux/macOS:         export HF_ENDPOINT=https://hf-mirror.com

用法：
    python scripts/download_model.py
"""
from pathlib import Path

import logging
import os

from huggingface_hub import snapshot_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_ID = "bert-base-chinese"
TARGET_DIR = Path(__file__).resolve().parent.parent / "models" / "bert-base-chinese"


def main() -> None:
    if not os.environ.get("HF_ENDPOINT"):
        logger.warning("未设置 HF_ENDPOINT，国内直连 HuggingFace 可能很慢；"
                       "建议先设置 HF_ENDPOINT=https://hf-mirror.com 再运行")
    if (TARGET_DIR / "model.safetensors").exists():
        logger.info(f"模型已存在于 {TARGET_DIR}，跳过下载")
        return
    logger.info(f"开始下载 {MODEL_ID} 到 {TARGET_DIR}（约 400MB）")
    snapshot_download(repo_id=MODEL_ID, local_dir=str(TARGET_DIR))
    logger.info("下载完成")


if __name__ == "__main__":
    main()
