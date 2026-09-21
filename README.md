# bert-Teams：BERT 文本分类实战（企业客服意图识别）

> 课程小组项目：围绕企业内部客服场景的 9 类意图分类，完成
> 「数据处理 → 多路线建模 → 评估 → 模型压缩 → Flask 部署」全流程。

## 一、项目目标

掌握文本分类项目的完整构建过程，覆盖四条技术路线并横向对比：

| 路线 | 技术 | 预期精度（test 集） | 特点 |
|------|------|------|------|
| 随机森林 | TF-IDF + RandomForest | ~0.83 | 传统机器学习基线，快速验证数据 |
| fasttext | 词级别有监督分类 | ~0.92 | 深度学习基线，毫秒级推理 |
| BERT | bert-base-chinese 微调 | ~0.98 | 主力模型（注意：数据为模板合成，分数偏高，见「数据质量」） |
| LLM | DeepSeek API + 系统提示词 | 零样本 | 零训练对照路线 |

在 BERT 基础上完成三种压缩优化并输出「精度-体积-延迟」对比：

| 压缩方式 | API / 方法 | 预期效果 |
|------|------|------|
| 动态量化 | `torch.quantization.quantize_dynamic`（仅 CPU） | 体积 → 约 40%，CPU 推理提速 |
| 知识蒸馏 | BERT → BiLSTM 学生（软标签，T=2，α=0.7） | 390MB → 约 25MB |
| 剪枝 | `prune.global_unstructured`（L1，30%） | 精度基本不掉，体积不变（逻辑剪枝） |

## 二、统一技术栈（全组必须一致）

| 项目 | 统一版本 | 说明 |
|------|------|------|
| Python | **3.11.x** | 用 uv 创建：`uv venv --python 3.11 .venv`（fasttext-wheel 仅支持到 3.11，全组统一） |
| 包管理 | uv | 依赖以 `pyproject.toml` 为准，改动需同步 `requirements.txt` |
| 深度学习 | torch 2.x + transformers 4.x | GPU 机器装 CUDA 版 torch（见下），其余机器 CPU 版即可 |
| 机器学习 | scikit-learn 1.x + jieba | RF 路线 |
| fasttext | fasttext-wheel 0.9.x | 在统一 3.11 环境中随 requirements 一起安装 |
| Web 部署 | **Flask 3.x** | 统一用 Flask，不用 FastAPI / Streamlit |
| 图表 | matplotlib | 混淆矩阵、对比曲线 |
| 配置管理 | `src/config.py` 单一入口 + `.env`（密钥） | 禁止在代码中散落路径常量、禁止提交密钥 |
| LLM | DeepSeek API（`DEEPSEEK_API_KEY` 存 `.env`） | 今晚必须注册拿 key |

## 三、十人分工

| # | 成员 | 模块 | 负责文件 | 交付物（答辩讲什么） | 跟踪 |
|---|------|------|------|------|------|
| 1 | **lingbai-i（组长）** | 统筹 | 全部 PR 审核 | 骨架整合、PPT 主线、答辩串词 | #11 |
| 2 | DaleFu22 | 数据加载 | `src/data.py` | 数据读取/清洗/Dataset 封装，讲 Dataset 三要素 | #12 |
| 3 | Evanjoy482 | 数据分析 | `outputs/eda/` | EDA 报告：类别/长度分布、模板规律与数据同源风险 | #13 |
| 4 | Hulk-Zhao | 随机森林 | `src/rf_baseline.py` | TF-IDF+RF 基线结果与体积/延迟 | #14 |
| 5 | jiangjn1996 | fasttext | `src/fasttext_baseline.py` | fasttext 训练与结果 | #15 |
| 6 | Jonny-Guo | BERT 训练 | `src/train.py` | 训练循环（14251 流程）、全量基线、训练曲线 | #16 |
| 7 | Mr-zhuyifan | BERT 评估 | `src/evaluate.py` | acc/F1/混淆矩阵/badcase 分析、人工评估集 | #17 |
| 8 | nahotoby341-hash | LLM 路线 | `src/llm_baseline.py` | 提示词设计、采样评估结果、延迟对比 | #18 |
| 9 | Twindy10 | 模型压缩 | `src/quantize.py` / `distill.py` / `prune.py` | 三件套对比表（精度/体积/延迟） | #19 |
| 10 | Xyafd | 部署展示 | `app/` | Flask 多模型对比页、现场 demo、PPT 制作 | #20 |

**任务领取方式**（每位成员）：
1. 打开仓库 Issues 页面，找到自己名下的 issue（#11~#20，与上表「跟踪」列对应）
2. 按 issue 里的「任务 / 验收标准 / 时间」执行，完成后在 issue 下回复结果截图或指标
3. 代码改动走个人分支 → PR → `dev`（流程见 [CONTRIBUTING.md](./CONTRIBUTING.md)），组长审核合入后关闭 issue

**全员必做**（人人动手）：
- 今晚各自跑通 mini 训练：`python -m src.train --subsample 5000 --epochs 1`，截图留存
- 每人写 10 条真实业务问句（覆盖各类别），交给评估组汇总进 `data/eval/test_queries.jsonl`

## 四、快速开始

```powershell
# 0. 创建环境（Python 3.11，全组统一）
uv venv --python 3.11 .venv
.\.venv\Scripts\Activate.ps1

# 1. 安装依赖
uv pip install -r requirements.txt
# GPU 机器追加（CUDA 版 torch，约 2.5GB，下载慢可挂机）：
uv pip install torch --index-url https://download.pytorch.org/whl/cu128 --force-reinstall

# 2. 准备模型（二选一）
python scripts/download_model.py        # 从镜像下载（先 set HF_ENDPOINT=https://hf-mirror.com）
# 或直接从组长处拷贝 models/bert-base-chinese/

# 3. mini 训练（全员跑通，几分钟）
python -m src.train --subsample 5000 --epochs 1 --output-dir checkpoints/mini

# 4. 全量训练（GPU 机器，约 30~60 分钟）
python -m src.train

# 5. 评估（合成测试集 + 人工评估集）
python -m src.evaluate --input data/test.txt --output-dir outputs/eval_test
python -m src.evaluate --input data/eval/test_queries.jsonl --output-dir outputs/eval_human

# 6. 压缩三件套
python -m src.quantize                  # 动态量化（CPU）
python -m src.distill                   # 蒸馏 BiLSTM 学生
python -m src.prune --amount 0.3        # 全局非结构化剪枝

# 7. 基线路线
python -m src.rf_baseline --subsample 20000     # RF 快速版
python -m src.fasttext_baseline                 # fasttext
python -m src.llm_baseline --sample 200         # 需先配置 .env

# 8. 部署
python -m app.flask_app --port 5000     # 浏览器打开 http://localhost:5000
```

## 五、模型产物规范（重要：避免"每人一个模型"的混乱）

| 模型产物 | 地位 | 说明 |
|------|------|------|
| 全量 `checkpoints/intent_bert/best.pt` | **唯一权威模型** | 组长机器全量训练（seed=42），答辩、压缩、部署全部以它为准 |
| 组员 mini 训练的 best.pt | 练手产物，不交付 | 只为跑通流程，精度无意义，**不要**用于任何正式环节 |
| `intent_bert_int8/best_int8.pt` | 从权威模型派生 | 量化组基于权威 best.pt 生成 |
| `intent_bilstm_distill/best.pt` | 从权威模型派生 | 蒸馏组以权威 best.pt 为教师 |
| `intent_bert_pruned/best.pt` | 从权威模型派生 | 剪枝组基于权威 best.pt 生成 |
| `rf/rf.pkl`、`fasttext/ft.bin` | 各自独立训练 | 基线路线，metrics.json 里写清训练配置即可 |

**产物共享方式**：模型不入库（.gitignore 已排除）。压缩、部署联调明早统一在组长机器上进行（该机器已有 `models/` 和权威 best.pt）；其他成员如需模型文件，由组长通过网盘/即时通讯发送 best.pt。

## 六、数据说明

- `data/train.txt` / `data/dev.txt` / `data/test.txt`：**已清洗的正式数据**，格式 `文本\t标签ID`
  （train 201,234 条 / dev 9,998 条 / test 11,095 条，9 类）
- `data/raw/{train,dev,test}.txt`：**原始未清洗语料**（202,500 / 10,000 / 11,099 条），归档用于对比与复现
- `data/synthetic_label_rules.json` + `scripts/clean_synthetic.py`：清洗规则表与清洗脚本
  （去重 1,266 条重复行；按 knowledge_base 重新分类 7 个双标签主题；报告见 `data/clean_reports/`）
- `data/class.txt` 9 个意图标签（8 业务 + 其他闲聊），行号即标签 ID
- `data/stopwords.txt` 749 个停用词（TF-IDF 用）
- `data/eval/test_queries.jsonl` 人工真实问句评估集（初始 18 条，评估组扩充至 118 条，覆盖全部 9 类，含 100 条全员新增问句）

**数据质量提示**：训练/测试集由模板合成，同源同分布，且原始语料存在 7 个主题被两个类别各标一半的
矛盾标注，因此未清洗测试集上的 98.4% 属于虚高。清洗后同一模型的分数为 98.40%，其中仍有 75.8% 的错误
来自团队内的口径分歧；真实水平以人工评估集（91.53%）为准——这是答辩的重要亮点而非问题。
完整分析见 [docs/eval_report.md](./docs/eval_report.md) 与 [data/eval/badcase_notes.md](./data/eval/badcase_notes.md)。

## 六、目录结构

```
bert-Teams/
├── src/
│   ├── config.py            # 全局配置单一入口
│   ├── data.py              # 数据加载 / Dataset / 子采样
│   ├── model.py             # BERT 分类器 / BiLSTM 学生模型
│   ├── train.py             # BERT 基线训练（--subsample 支持 mini 训练）
│   ├── evaluate.py          # 评估：acc/F1/混淆矩阵/badcase
│   ├── rf_baseline.py       # 随机森林路线
│   ├── fasttext_baseline.py # fasttext 路线
│   ├── llm_baseline.py      # LLM 提示词路线
│   ├── quantize.py          # 动态量化
│   ├── distill.py           # 知识蒸馏
│   ├── prune.py             # 全局非结构化剪枝
│   └── predict.py           # 统一推理封装（部署层调用）
├── app/
│   ├── flask_app.py         # Flask 服务（/api/predict 多模型）
│   └── templates/index.html # 测试页面（响应时间 + 清空 + 多模型切换）
├── scripts/download_model.py
├── data/  models/  checkpoints/  outputs/   # 后三者不入库
└── pyproject.toml / requirements.txt / .env.example
```

## 七、时间轴（明天下午答辩）

| 时间 | 事项 | 负责人 |
|------|------|--------|
| 今晚 1h | 全组会：分工确认、环境安装、LLM 组注册 DeepSeek 拿 key | 全员 |
| 今晚 | 全员 mini 训练跑通截图；数据组出 EDA；RF/fasttext 开跑 | 各模块 |
| 今晚睡前 | 组长机器挂机全量 BERT 训练 | 训练组 |
| 明早 | 压缩三件套 + LLM 采样评估 + 人工评估集汇总 + 评估报告 | 各模块 |
| 明午 | Flask 联调、结果合入 PPT、串讲演练 + 互相提问 | 组长+部署组 |
| 止损线 | 明早 11:00 压缩/LLM 没结果就降级为「方案+初步实验」，不硬撑 | 组长 |

## 八、答辩故事线

1. 业务场景与数据介绍（9 类企业客服意图，20 万条）
2. 四条路线横向对比表（精度/体积/延迟）——先 RF/fasttext 基线，再 BERT，再 LLM
3. 数据质量发现：模板合成数据分数虚高 → 自建人工评估集验证真实水平
4. 模型压缩：量化/蒸馏/剪枝三件套对比，给出「CPU 部署选蒸馏 BiLSTM」的结论
5. Flask 现场演示多模型对比
6. 每人讲自己负责的模块（评委必问：你负责的哪部分？遇到什么问题？怎么优化？）

## 九、协作纪律

- 分支流程见 [CONTRIBUTING.md](./CONTRIBUTING.md)：个人分支 → PR → `dev`
- 代码/注释规范见 [Agent.md](./Agent.md)：中文注释、文件级 docstring、固定随机种子
- 参考项目与课程代码只许理解后重写，禁止整段复制；关键模块作者必须能脱稿讲解
- commit message 用简体中文 Conventional Commits 格式
