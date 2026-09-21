# BERT 意图识别（bert-Teams）

基于 `bert-base-chinese` 微调的中文意图识别模型，面向企业内部客服场景，
识别 9 类意图（考勤政策 / IT 支持 / 财务报销 / 人事入职 / 行政后勤 / 工程设备 /
法务合同 / 销售市场 / 闲聊）。当模型对所有类别的置信度都低于阈值时，
判定为闲聊并拒绝回答（可覆盖超出范围的敏感问题）。

## 数据

- `data/class.txt`：9 个类别名，行号即标签编号（0-8）。
- `data/train.txt`：训练集，20.25 万条，制表符分隔（文本\t标签编号），9 类均衡。
- `data/dev.txt`：验证集，1 万条，仅含类别 0-7。
- `data/test.txt`：测试集，1.1 万条，9 类齐全。
- `data/intent_train.csv`：480 条模板种子数据（与 train.txt 部分重叠）。
- `data/eval/test_queries.jsonl`：评测查询，含 `out_of_scope` 越界样本。

## 环境

使用 conda 环境 `pytorch_gpu`（torch 2.5.1 + CUDA，transformers 5.16）。

```bash
conda activate pytorch_gpu
# 首次下载模型需使用镜像（本机无法直连 huggingface.co）
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

## 训练

```bash
python -m src.train
```

输出：最优模型保存至 `models/bert-intent/`，训练日志导出至 `outputs/train_log.json`。

## 评估

```bash
python -m src.evaluate
```

输出：`outputs/report.md`、混淆矩阵与阈值曲线 PNG。

## 推理

```bash
# 单条预测
python -m src.infer "公司上下班时间是几点"

# 交互模式
python -m src.infer
```

拒答规则：最高类别置信度低于阈值（默认 0.5，见 `src/config.py`）时判定为闲聊并拒答；
预测意图本身为闲聊类时同样不提供业务回答。

## LLM 基线（对照路线）

调用 DeepSeek 大模型做零样本意图分类，与 BERT 微调路线对照。

### 环境准备

1. 注册 DeepSeek 开放平台（https://platform.deepseek.com）并创建 API Key；
2. 按 `.env.example` 复制为 `.env`（已被 .gitignore 忽略，禁止提交），填入 `DEEPSEEK_API_KEY`。

### 运行评估

```bash
python -m src.llm_baseline --sample 200        # 补齐所有未评估的提示词版本
python -m src.llm_baseline --sample 200 --version v3   # 只评估指定版本
```

结果写入 `outputs/llm/metrics.json`（含各版本准确率、平均延迟与逐条记录）；
提示词迭代过程见 `outputs/llm/prompt_iterations.md`；答辩讲解点见
`outputs/llm/presentation_notes.md`。

### 当前结果（200 条采样，固定种子 42）

| 版本 | 准确率 | 平均延迟 |
| --- | --- | --- |
| v1 基础版（零样本） | 0.7800 | 897.0 ms |
| v2 优化版（易混淆规则） | 0.8050 | 807.4 ms |
| v3 优化版（扩展人事定义+修正规则） | 0.8800 | 761.5 ms |

两版提示词迭代共提升 10.0 个百分点；延迟差异主要来自网络波动。

