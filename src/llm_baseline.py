"""LLM 路线：通过远程大模型 API + 系统提示词完成意图分类。

不训练任何参数，靠提示词约束模型输出 9 类标签之一，作为零样本对照路线。
默认使用 DeepSeek（OpenAI 兼容接口），密钥从 .env 的 DEEPSEEK_API_KEY 读取。

用法：
    python -m src.llm_baseline --input data/test.txt --sample 200
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

import argparse
import json
import logging
import os
import random
import time

import requests

from src.config import CLASS_FILE, OUTPUT_DIR, SEED, TEST_FILE
from src.data import load_labels, load_txt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

API_URL = "https://api.deepseek.com/chat/completions"
API_MODEL = "deepseek-chat"

# 各类别的中文释义，帮助模型理解标签边界（顺序须与 class.txt 一致）
LABEL_DESC = {
    "policy_attendance": "考勤制度：上下班、打卡、请假、年假、加班、远程办公",
    "it_vpn": "IT 支持：VPN、密码、电脑、打印机、邮箱、网络",
    "finance_expense": "财务报销：报销、发票、工资、差旅、借支",
    "hr_onboarding": "人力资源：入职、离职、社保、福利、内推",
    "admin_logistics": "行政后勤：食堂、班车、宿舍、会议室、活动",
    "engineering_eq": "工程设备：产线、维修、点检、备件、计量校准",
    "legal_contract": "法务合同：合同、公章、保密、知识产权、合规",
    "sales_marketing": "销售市场：报价、订单、客户、展会、预测",
    "other_chitchat": "其他/闲聊：问候、闲聊、与业务无关的问题",
}

SYSTEM_PROMPT = """# 角色
你是企业内部客服场景的意图分类助手。

# 任务
阅读用户输入的一句话，从以下 9 个类别中选出唯一最匹配的标签。

# 类别定义（严格限定，不得新增）
{categories}

# 规则
1. 只输出类别标签名本身（英文），不要输出解释、标点或换行。
2. 与公司业务无关的问候、闲聊一律输出 other_chitchat。
3. 难以判断时选择最接近的类别，禁止拒答。"""


def build_prompt() -> str:
    """按 class.txt 当前标签体系动态生成系统提示词。"""
    categories = "\n".join(f"- {name}：{LABEL_DESC.get(name, name)}"
                           for name in load_labels(CLASS_FILE))
    return SYSTEM_PROMPT.format(categories=categories)


def classify(text: str, api_key: str, timeout: int = 30) -> Optional[str]:
    """调用远程大模型对单条文本分类。

    参数:
        text: 待分类文本。
        api_key: DeepSeek API 密钥。
        timeout: 单次请求超时秒数。

    返回:
        解析出的标签名；响应中找不到合法标签时返回 None。

    异常:
        requests.HTTPError: HTTP 层错误（4xx/5xx）直接抛出，便于暴露配额/密钥问题。
    """
    resp = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": API_MODEL,
            "messages": [
                {"role": "system", "content": build_prompt()},
                {"role": "user", "content": text},
            ],
            "temperature": 0,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"].strip()
    label_names = load_labels(CLASS_FILE)
    if content in label_names:
        return content
    # 模型未严格遵守输出格式时，退化到子串匹配提取标签
    for name in label_names:
        if name in content:
            return name
    return None


def main(args) -> None:
    load_dotenv()
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("未找到 DEEPSEEK_API_KEY，请在 .env 中配置（参照 .env.example）")

    texts, labels = load_txt(Path(args.input))
    if args.sample and args.sample < len(texts):
        idx = random.Random(SEED).sample(range(len(texts)), args.sample)
        texts = [texts[i] for i in idx]
        labels = [labels[i] for i in idx]
    label_names = load_labels(CLASS_FILE)
    logger.info(f"评估样本数={len(texts)}，并发={args.workers}")

    def _one(pair):
        text, gold = pair
        t0 = time.perf_counter()
        pred = classify(text, api_key)
        return gold, pred, (time.perf_counter() - t0) * 1000

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, r in enumerate(pool.map(_one, zip(texts, labels))):
            results.append(r)
            if (i + 1) % 20 == 0:
                logger.info(f"已完成 {i + 1}/{len(texts)}")

    valid = [(g, p) for g, p, _ in results if p is not None]
    acc = sum(1 for g, p in valid if label_names[g] == p) / max(len(valid), 1)
    avg_latency = sum(lat for _, _, lat in results) / max(len(results), 1)
    failed = len(results) - len(valid)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps({
        "args": vars(args),
        "samples": len(texts), "accuracy": acc,
        "avg_latency_ms": avg_latency, "parse_failed": failed,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"LLM 路线: acc={acc:.4f} 平均延迟={avg_latency:.0f}ms "
                f"解析失败={failed} 条，结果已写入 {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="LLM 零样本意图分类评估")
    ap.add_argument("--input", default=str(TEST_FILE), help=".txt 数据文件")
    ap.add_argument("--sample", type=int, default=200,
                    help="随机采样条数（控制 API 调用量），0 为全量")
    ap.add_argument("--workers", type=int, default=4, help="并发请求数")
    ap.add_argument("--output-dir", default=str(OUTPUT_DIR / "llm"))
    main(ap.parse_args())
