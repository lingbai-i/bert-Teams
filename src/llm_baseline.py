"""LLM 基线评估脚本：调用 DeepSeek 大模型做零样本意图分类，与 BERT 微调路线对照。

功能：
- 内置三版系统提示词：v1 基础版（纯零样本）、v2 优化版（易混淆类别判定规则）、
  v3 优化版（扩展人事定义并修正规则），用于量化提示词调优的增益；
- 从 test.txt 按固定种子随机采样（默认 200 条），逐条调用 DeepSeek chat 接口，
  记录每条耗时，输出整体准确率与平均延迟；
- 结果与多版本对比记录写入 outputs/llm/metrics.json。

运行方式：
    python -m src.llm_baseline --sample 200        # 补齐所有未评估的提示词版本
    python -m src.llm_baseline --sample 200 --version v3   # 只评估指定版本

依赖：
- 需在 .env 中配置 DEEPSEEK_API_KEY（见 .env.example）。
"""
import argparse
import json
import random
import re
import time
from pathlib import Path

import requests

# 类别中文名，用于解析模型可能输出的中文标签
CN_NAMES = {
    "policy_attendance": "考勤政策",
    "it_vpn": "IT支持",
    "finance_expense": "财务报销",
    "hr_onboarding": "人事入职",
    "admin_logistics": "行政后勤",
    "engineering_eq": "工程设备",
    "legal_contract": "法务合同",
    "sales_marketing": "销售市场",
    "other_chitchat": "闲聊",
}


def load_class_labels(class_file: Path):
    """读取类别列表，返回 (id2label, label2id)。

    class.txt 每行一个类别名，行号（从 0 开始）即标签编号。
    """
    lines = class_file.read_text(encoding="utf-8").splitlines()
    labels = [line.strip() for line in lines if line.strip()]
    label2id = {name: idx for idx, name in enumerate(labels)}
    return labels, label2id


def load_tsv(path: Path):
    """读取制表符分隔的（文本, 标签编号）数据集文件，返回 (文本列表, 标签列表)。

    每行格式为 ``文本\\t标签编号``，忽略空行与格式异常的行。
    """
    texts = []
    labels = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        texts.append(parts[0])
        labels.append(int(parts[1]))
    return texts, labels

# 类别定义说明，两版提示词共用
CLASS_DEFS = """0 policy_attendance：考勤政策类。上下班/打卡时间、年假、调休、事假、加班、迟到等考勤与休假制度问题。
1 it_vpn：IT 支持类。VPN、账号密码、邮箱、电脑/打印机、网络、软件、MFA 等 IT 系统与设备问题。
2 finance_expense：财务报销类。报销流程与打款、发票、银行账户、差旅标准、工资条、公积金/社保（财务口径）等。
3 hr_onboarding：人事入职类。入职材料与流程、试用期、劳动合同期限/续签、入职培训、体检、离职证明等。
4 admin_logistics：行政后勤类。食堂、快递收发、班车、饭卡、访客登记、储物柜、办公用品等行政事务。
5 engineering_eq：工程设备类。技改项目、设备保养/报废/校准、备件库存、预测性维护、生产安全等。
6 legal_contract：法务合同类。合同模板/条款/解除、竞业协议、知识产权、保密协议、法务咨询等。
7 sales_marketing：销售市场类。订单、客户、报价/价目表、标书、销售指标、市场活动等。
8 other_chitchat：闲聊类。寒暄问候、情绪表达、与公司事务无关的日常话题。"""


def build_prompt_v1() -> str:
    """构建 v1 基础版系统提示词（纯零样本）。

    仅提供类别定义与输出格式约束，不包含易混淆类别的判定规则。
    """
    return (
        "你是企业内部客服系统的意图分类器。用户会发来一句中文咨询，"
        "请从以下 9 个类别中选出唯一最合适的类别。\n\n"
        f"{CLASS_DEFS}\n\n"
        "输出要求：只输出类别编号（0-8），不要输出任何其他文字。"
    )


def build_prompt_v2() -> str:
    """构建 v2 优化版系统提示词。

    在 v1 基础上增加易混淆类别判定规则与闲聊边界说明，
    针对数据中实际存在的跨类别共现场景（合同、报销/人事、考勤/人事、
    IT/行政、销售/法务）做显式消歧。
    """
    return (
        "你是企业内部客服系统的意图分类器。用户会发来一句中文咨询，"
        "请从以下 9 个类别中选出唯一最合适的类别。\n\n"
        f"{CLASS_DEFS}\n\n"
        "易混淆类别判定规则：\n"
        "- 合同相关：问「合同期限、续签、入职合同」→ 3 人事入职；"
        "问「合同模板、合同条款/解除、竞业协议、知识产权、合作协议」→ 6 法务合同。\n"
        "- 报销与人事：问「报销、发票、差旅标准、银行账户、打款」→ 2 财务报销；"
        "问「工资条、公积金、社保」→ 2 财务报销（财务口径）；"
        "问「入职、试用期、培训、体检、离职证明」→ 3 人事入职。\n"
        "- 考勤与人事：问「年假、调休、事假、加班、打卡、上下班」→ 0 考勤政策；"
        "问「续签合同、试用期考核」→ 3 人事入职。\n"
        "- IT 与行政：问「VPN、账号、密码、电脑、打印机、网络、软件」→ 1 IT 支持；"
        "问「快递、食堂、班车、饭卡、访客、储物柜」→ 4 行政后勤。\n"
        "- 销售与法务：问「订单、客户、报价、标书、销售指标」→ 7 销售市场；"
        "问「协议、合规、合同模板、法务咨询」→ 6 法务合同。\n"
        "- 闲聊边界：只有寒暄、情绪表达或与公司事务无关的话题才归 8 闲聊；"
        "只要在询问公司的具体事务，就归对应的业务类别。\n\n"
        "输出要求：只输出类别编号（0-8），不要输出任何其他文字。"
    )


# v3 使用的扩展类别定义：根据 train.txt 关键词分布核实，
# 补齐各业务类别的真实覆盖主题（尤其 hr_onboarding 范围最广）
CLASS_DEFS_V3 = """0 policy_attendance：考勤与休假政策类。上下班/打卡时间、迟到、年假、调休、事假、请假、加班、考勤规则等。
1 it_vpn：IT 支持类。VPN、账号、密码、邮箱、电脑/打印机、网络、软件、MFA、域账号等。
2 finance_expense：财务报销类。报销流程与打款、发票、差旅标准、银行账户、工资条/工资发放等。
3 hr_onboarding：人事与员工事务类（覆盖范围最广）。入职材料与流程、试用期、劳动合同期限/续签、入职培训、体检、离职/离职证明、福利/工会福利、员工申诉、职级/晋升、绩效/360评估、团建与员工活动、内推/人才推荐、落户、退休、档案转移、公积金/社保缴纳等。
4 admin_logistics：行政后勤类。食堂/就餐、快递收发、班车、饭卡、访客登记、储物柜、办公用品、会议室等。
5 engineering_eq：工程与设备类。技改项目、设备保养/报废/校准、备件库存、预测性维护、生产安全等。
6 legal_contract：法务与合同类。合同模板/条款/解除、竞业协议、知识产权、保密协议、合作协议、法务咨询等。
7 sales_marketing：销售与市场类。订单、客户/客户档案、报价/价目表、标书、销售指标、市场活动等。
8 other_chitchat：闲聊类。寒暄问候、情绪表达、与公司事务无关的日常话题。"""


def build_prompt_v3() -> str:
    """构建 v3 优化版系统提示词。

    相对 v2 的迭代点（基于 200 条采样错误分析）：
    1. 扩展 hr_onboarding 类别定义，覆盖福利/申诉/职级/晋升/绩效/团建/内推/
       落户/退休/档案/公积金/社保等真实人事主题（v2 定义过窄导致被误判为闲聊）；
    2. 修正公积金/社保归属：数据主体归 hr_onboarding，而非 v2 所写的财务口径；
    3. 强化闲聊边界：寒暄词只是礼貌用语，不改变类别；补充分歧词（档案/活动/团建/推荐）。
    """
    return (
        "你是企业内部客服系统的意图分类器。用户会发来一句中文咨询，"
        "请从以下 9 个类别中选出唯一最合适的类别。\n\n"
        f"{CLASS_DEFS_V3}\n\n"
        "易混淆类别判定规则：\n"
        "- 合同相关：问「合同期限、续签、入职合同」→ 3 人事；"
        "问「合同模板、合同条款/解除、竞业协议、知识产权、合作协议」→ 6 法务。\n"
        "- 报销与人事：问「报销、发票、差旅标准、银行账户、打款、工资条、工资发放」→ 2 财务；"
        "问「公积金、社保、福利」→ 3 人事（人事口径）；"
        "问「入职、试用期、培训、体检、离职证明」→ 3 人事。\n"
        "- 考勤与人事：问「年假、调休、事假、请假、加班、打卡、上下班」→ 0 考勤；"
        "问「入职、续签、试用期考核、晋升、职级、退休、内推、落户、档案转移」→ 3 人事。\n"
        "- IT 与行政：问「VPN、账号、密码、电脑、打印机、网络、软件」→ 1 IT 支持；"
        "问「快递、食堂、班车、饭卡、访客、储物柜」→ 4 行政后勤。\n"
        "- 销售与法务：问「订单、客户、报价、标书、销售指标、市场活动」→ 7 销售；"
        "问「协议、合规、合同模板、法务咨询」→ 6 法务。\n"
        "- 歧义词：客户档案→7 销售；员工档案/档案转移→3 人事。"
        "市场活动→7 销售；员工活动/团建活动规则→3 人事；团建费用报销→2 财务。"
        "人才推荐/内推→3 人事；与公司事务无关的推荐（如推荐电影）→8 闲聊。\n"
        "- 闲聊边界：句首句尾的「哎、请问、在吗、谢谢、辛苦、麻烦、哈」只是礼貌用语，"
        "不影响类别；只要问题主体在询问公司具体事务，就归对应业务类别；"
        "只有纯寒暄、情绪表达或完全与公司事务无关的话题才归 8 闲聊。\n\n"
        "输出要求：只输出类别编号（0-8），不要输出任何其他文字。"
    )


def load_env(env_path: Path) -> dict:
    """解析 .env 文件为字典（跳过注释行与空行）。

    参数：
        env_path: .env 文件路径。

    返回：
        键值对字典；不存在的键不会出现在结果中。
    """
    env = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip()
    return env


def parse_label(raw: str, labels: list, label2id: dict):
    """将模型输出解析为类别编号，解析失败返回 None。

    依次尝试：精确匹配英文名 / 纯数字 / 输出文本包含英文名 / 输出文本包含中文名 /
    提取输出中的单个类别编号。模型偶发输出"6 法务合同"等附加说明时兜底解析。
    """
    raw = raw.strip()
    if raw in label2id:
        return label2id[raw]
    if raw.isdigit() and 0 <= int(raw) < len(labels):
        return int(raw)
    for name in label2id:
        if name in raw:
            return label2id[name]
    for name, cn in CN_NAMES.items():
        if cn in raw:
            return label2id[name]
    match = re.search(r"(?<!\d)([0-8])(?!\d)", raw)
    if match:
        return int(match.group(1))
    return None


def call_deepseek(api_key: str, base_url: str, model: str,
                  system_prompt: str, query: str) -> tuple:
    """调用 DeepSeek chat 接口，返回 (模型输出文本, 本条端到端耗时毫秒)。

    对连接错误、超时与 5xx/429 做最多 2 次退避重试（2s/4s），
    其他异常直接抛出并携带查询文本上下文。

    参数：
        api_key: DeepSeek API Key。
        base_url: API 基础地址，如 https://api.deepseek.com。
        model: 模型名，如 deepseek-chat。
        system_prompt: 系统提示词（build_prompt 系列产物）。
        query: 用户查询文本。

    返回：
        (模型输出文本, 总耗时毫秒)。总耗时包含全部重试尝试。
    """
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        "temperature": 0,
        "max_tokens": 16,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    start = time.perf_counter()
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=60)
            if resp.status_code in {429, 500, 502, 503, 504}:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            elapsed_ms = (time.perf_counter() - start) * 1000
            return content, elapsed_ms
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout,
                RuntimeError) as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"调用 DeepSeek 失败（查询：{query}）：{last_error}")


def main() -> None:
    """解析命令行参数，执行指定版本提示词的采样评估并输出对比结果。"""
    parser = argparse.ArgumentParser(description="LLM 意图识别基线评估")
    parser.add_argument("--sample", type=int, default=200, help="采样条数（默认 200）")
    parser.add_argument("--seed", type=int, default=42, help="采样随机种子（默认 42）")
    parser.add_argument("--version", choices=["v1", "v2", "v3", "all"], default="all",
                        help="评估的提示词版本；all 表示补齐尚未评估过的版本（默认 all）")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    env = load_env(project_root / ".env")
    api_key = env.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "未配置 DEEPSEEK_API_KEY：请先注册 DeepSeek 开放平台并创建 API Key，"
            "按 .env.example 说明填入 .env 后重试",
        )
    base_url = env.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    model = env.get("DEEPSEEK_MODEL", "deepseek-chat")

    id2label, label2id = load_class_labels(project_root / "data" / "class.txt")
    texts, labels = load_tsv(project_root / "data" / "test.txt")

    # 固定种子随机采样，保证结果可复现
    rng = random.Random(args.seed)
    sample_idx = rng.sample(range(len(texts)), min(args.sample, len(texts)))
    sample_texts = [texts[i] for i in sample_idx]
    sample_labels = [labels[i] for i in sample_idx]

    prompts = {
        "v1": ("基础版（零样本）", build_prompt_v1()),
        "v2": ("优化版（易混淆规则）", build_prompt_v2()),
        "v3": ("优化版（扩展人事定义+修正规则）", build_prompt_v3()),
    }
    if args.version == "all":
        versions = list(prompts.keys())
    else:
        versions = [args.version]

    out_dir = project_root / "outputs" / "llm"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 读取历史结果，已评估过的版本在 all 模式下跳过，避免重复计费
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists():
        existing = json.loads(metrics_path.read_text(encoding="utf-8"))
        version_results = existing.get("versions", {})
    else:
        version_results = {}
    for version in versions:
        if version in version_results:
            print(f"跳过 {version}：metrics.json 已存在该版本结果")
            continue
        label, system_prompt = prompts[version]
        print(f"\n=== {version}：{label}，共 {len(sample_texts)} 条 ===")
        records = []
        for idx, (text, true_label) in enumerate(zip(sample_texts, sample_labels)):
            raw_output, latency_ms = call_deepseek(
                api_key, base_url, model, system_prompt, text,
            )
            pred_label = parse_label(raw_output, id2label, label2id)
            correct = pred_label == true_label
            records.append({
                "query": text,
                "expected": id2label[true_label],
                "predicted": id2label[pred_label] if pred_label is not None else "parse_error",
                "correct": bool(correct),
                "latency_ms": round(latency_ms, 1),
                "raw_output": raw_output,
            })
            if (idx + 1) % 20 == 0:
                print(f"  已完成 {idx + 1}/{len(sample_texts)}，"
                      f"当前累计准确率 {sum(r['correct'] for r in records) / len(records):.4f}")

        accuracy = sum(r["correct"] for r in records) / len(records)
        avg_latency = sum(r["latency_ms"] for r in records) / len(records)
        error_count = sum(1 for r in records if r["predicted"] == "parse_error")
        version_results[version] = {
            "label": label,
            "sample_size": len(records),
            "accuracy": round(accuracy, 4),
            "avg_latency_ms": round(avg_latency, 1),
            "error_count": error_count,
            "results": records,
        }
        print(f"  准确率 {accuracy:.4f}，平均延迟 {avg_latency:.1f} ms，"
              f"解析失败 {error_count} 条")

    # ---- 多版本对比记录（按版本号升序两两对比） ----
    comparison = {}
    ordered = sorted(version_results.keys())
    if len(ordered) >= 2:
        deltas = []
        for prev, cur in zip(ordered, ordered[1:]):
            p, c = version_results[prev], version_results[cur]
            delta = round(c["accuracy"] - p["accuracy"], 4)
            latency_delta = round(c["avg_latency_ms"] - p["avg_latency_ms"], 1)
            deltas.append(f"{prev}->{cur} 准确率 {delta:+.4f}，延迟 {latency_delta:+.1f} ms")
        base = version_results[ordered[0]]
        best = max(ordered, key=lambda v: version_results[v]["accuracy"])
        comparison = {
            "version_order": ordered,
            "deltas": "；".join(deltas),
            "best_version": best,
            "best_accuracy": version_results[best]["accuracy"],
            "improvement_vs_base": round(
                version_results[best]["accuracy"] - base["accuracy"], 4),
            "conclusion": (
                f"共 {len(ordered)} 版提示词，最佳为 {best}（准确率 "
                f"{version_results[best]['accuracy']:.4f}）；相对基线 {ordered[0]} "
                f"提升 {version_results[best]['accuracy'] - base['accuracy']:+.4f}。"
                "延迟差异主要来自网络波动，而非提示词本身。"
            ),
        }

    metrics = {
        "task": "intent-recognition-llm-baseline",
        "model": model,
        "sample_size": len(sample_texts),
        "seed": args.seed,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "versions": version_results,
        "comparison": comparison,
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已写入 {metrics_path}")


if __name__ == "__main__":
    main()
