"""推理脚本：加载训练好的模型，对单条或批量文本做意图识别。

命令行用法：
    python -m src.infer "查询文本"            # 预测单条
    python -m src.infer                       # 进入交互模式，q 退出

判定规则：最高类别置信度低于拒答阈值时判定为闲聊并拒绝回答；
预测意图为闲聊类时同样拒绝提供业务回答。

依赖关系：
- 依赖 src.predictor；
- 通过 `python -m src.infer` 运行。
"""
import json
import sys
from typing import List

from src.config import Config
from src.predictor import IntentPredictor


def print_result(result: dict) -> None:
    """以易读格式打印单条预测结果。"""
    action = "拒答" if result["rejected"] else "回答"
    print(json.dumps({
        "查询": result["text"],
        "预测意图": result["intent"],
        "置信度": round(result["confidence"], 4),
        "行为": action,
        "说明": result["reason"],
    }, ensure_ascii=False, indent=2))


def main() -> None:
    """解析命令行参数并执行推理。"""
    cfg = Config()
    predictor = IntentPredictor(threshold=cfg.reject_threshold)
    args = sys.argv[1:]

    if args:
        queries: List[str] = args
    else:
        queries = []
        print("进入交互模式（输入 q 或 exit 退出）")
        while True:
            try:
                query = input("> ").strip()
            except EOFError:
                break
            if query.lower() in {"q", "exit", "quit"}:
                break
            if query:
                queries.append(query)
                print_result(predictor.predict(query))
        return

    for query in queries:
        print_result(predictor.predict(query))


if __name__ == "__main__":
    main()
