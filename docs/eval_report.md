# BERT 意图分类评估报告（合成集 vs 人工集）

> 模块负责人：Mr-zhuyifan（README 分工表 #7：BERT 评估 / src/evaluate.py）
> 评估对象：`models/best.pt`（组长的全量训练权重，seed=42，21128 词表 bert-base-chinese 微调）
> 文中所有数字均由脚本产出，可用第七节命令完整复现。

## 一、结论摘要（答辩用）

| 指标 | 合成测试集 | 人工评估集 | 差值（人工 - 合成） |
|------|-----------|-----------|---------------------|
| 样本数 | 11099 | 118 | -10981 |
| accuracy | 0.9840 | 0.8898 | **-9.42pp** |
| macro-F1 | 0.9841 | 0.8856 | **-9.85pp** |
| bad case 数 | 178 | 13 | -165 |
| 错误率 | 1.60% | 11.02% | +9.42pp |

三条结论：

1. **9 个类别在人工集上全部退化**，退化最重的是 `other_chitchat`（-21.74pp）、
   `admin_logistics`（-18.52pp）、`legal_contract`（-16.77pp）。
2. **合成集的 178 条错误 100% 可归因到数据本身**：双标签主题 135 条（75.8%）
   + 英文缩写捷径 43 条（24.2%），没有一条属于"模型没学会"。
3. **人工集的 13 条错误 100% 可归因到标签体系**：越界拒答类缺失 + 部门边界模糊，
   且其中 8 条置信度 >= 0.999，属于"高置信度错误"而非"拿不准"。

因此：**合成集的 98.4% 衡量的是"模型拟合同源模板的能力"，人工集的 88.98% 才是真实语义边界下的可用性；
9.4 个百分点的差距是数据分布差距，不是模型能力差距。**

## 二、评估设置

| 项 | 内容 |
|------|------|
| 模型 | `models/best.pt`（390MB，201 个张量，`load_state_dict` 严格匹配，missing/unexpected 均为 0） |
| 分词器 / 骨干 | `models/bert-base-chinese`（vocab.txt + tokenizer.json + config.json） |
| 合成集 | `data/test.txt`，11099 条，`文本\t标签ID`，9 类各 615~747 条 |
| 人工集 | `data/eval/test_queries.jsonl`，118 条，`{"query","intent"}` |
| 截断长度 | max_len = 48（`src/config.py`） |
| 批次 / 设备 | bs = 64 / CUDA（RTX 3060 Laptop） |
| 指标 | accuracy、macro-F1、weighted-F1、逐类 precision/recall/F1、混淆矩阵 |

## 三、合成测试集结果

```
accuracy = 0.9840   macro_f1 = 0.9841   weighted_f1 = 0.9839
```

| 类别 | precision | recall | F1 | support |
|------|-----------|--------|-----|---------|
| policy_attendance | 0.9983 | 0.9400 | 0.9683 | 1250 |
| it_vpn | 1.0000 | 0.9992 | 0.9996 | 1250 |
| finance_expense | 0.9858 | 0.9984 | 0.9921 | 1250 |
| hr_onboarding | 0.9551 | 0.9704 | 0.9627 | 1250 |
| admin_logistics | 1.0000 | 1.0000 | 1.0000 | 1250 |
| engineering_eq | 0.9667 | 1.0000 | 0.9831 | 1250 |
| legal_contract | 0.9650 | 0.9704 | 0.9677 | 1250 |
| sales_marketing | 0.9887 | 0.9792 | 0.9839 | 1250 |
| other_chitchat | 1.0000 | 1.0000 | 1.0000 | 1099 |

178 条错误的构成（`outputs/eval_test/badcases.txt`）：

| 混淆对（真实 -> 预测） | 条数 | 占错误比 | 归因 |
|------------------------|------|----------|------|
| policy_attendance -> hr_onboarding | 57 | 32.0% | 病假工资/育儿假/陪产假在两类间 50:50 标注 |
| hr_onboarding -> legal_contract | 37 | 20.8% | 保密协议/竞业协议在两类间 50:50 标注 |
| legal_contract -> engineering_eq | 23 | 12.9% | NDA 触发"英文字母串 -> engineering_eq"捷径 |
| sales_marketing -> engineering_eq | 19 | 10.7% | NPS 同上 |
| policy_attendance -> finance_expense | 18 | 10.1% | 加班费在两类间 51:49 标注 |
| 其余 4 对 | 24 | 13.5% | 合同模板双向混淆 + 缩写捷径 |

## 四、人工评估集结果

### 4.1 数据集扩充（18 -> 118 条）

- 原有 18 条（4 类各 4 条 + 2 条越界）保留原句不变。
- 本次按"全员每人 10 条真实问句"补入 **100 条**，覆盖此前为空的 4 个类别
  （admin_logistics、engineering_eq、legal_contract、sales_marketing）。
- 汇总后 118 条，类别分布 12~14 条/类，**去重检查无重复句**（`query` 完全相同计为重复）。
- 标签对齐：原文件 2 条使用 `out_of_scope`，该标签名不在 `data/class.txt` 中
  （第 9 类为 `other_chitchat`），已归一到 `other_chitchat`，否则 evaluate 会直接抛
  "未知标签名"而无法评估。
- 问句写法刻意贴合真实员工口径（口语、省略、带具体时间/系统名，如"上个月""六楼""E023"），
  与合成集的"主题词 + 模板后缀"句式不同：人工句长中位数 12 字，合成句 9 字。

### 4.2 结果

```
accuracy = 0.8898   macro_f1 = 0.8856   weighted_f1 = 0.8865
```

| 类别 | precision | recall | F1 | support |
|------|-----------|--------|-----|---------|
| policy_attendance | 0.8750 | 1.0000 | 0.9333 | 14 |
| it_vpn | 0.9286 | 0.9286 | 0.9286 | 14 |
| finance_expense | 0.9286 | 0.9286 | 0.9286 | 14 |
| hr_onboarding | 0.8235 | 1.0000 | 0.9032 | 14 |
| admin_logistics | 0.7333 | 0.9167 | 0.8148 | 12 |
| engineering_eq | 1.0000 | 0.9167 | 0.9565 | 12 |
| legal_contract | 1.0000 | 0.6667 | 0.8000 | 12 |
| sales_marketing | 0.8571 | 1.0000 | 0.9231 | 12 |
| other_chitchat | 1.0000 | 0.6429 | 0.7826 | 14 |

13 条错误分三类：

1. **第 9 类缺"越界/拒答"语义（5 条）**：`other_chitchat` recall 仅 0.6429。
   训练语料中该类全是纯闲聊，模型学到"闲聊词 -> other_chitchat"，句子一沾业务词就被拉走：
   别人的考勤记录能查吗 -> policy_attendance（置信度 1.000）、能帮我查下同事这个月的考勤吗 -> policy_attendance（1.000）、
   CEO 的工资是多少 -> finance_expense（0.542）、公司CEO的年薪是多少 -> hr_onboarding（0.477）、
   帮我给宠物起个名字 -> admin_logistics（1.000）。
2. **部门边界本身模糊（6 条）**：`legal_contract` recall 0.6667，
   收集员工个人信息要不要签同意书 -> hr_onboarding、劳动合同到期不续签需要提前通知吗 -> hr_onboarding、
   我从网上下载的开源代码能商用吗 -> it_vpn、客户要签保密协议用公司模板还是对方的 -> sales_marketing；
   另有会议室的投影仪连不上我的笔记本（it_vpn -> admin_logistics）、
   项目聚餐的费用走哪个科目（finance_expense -> admin_logistics）。
   这类问句标注时人人都能讲出理由，属于**单标签分类的粒度天花板**，答辩时应说明裁定口径（按主要意图归属）。
3. **`admin_logistics` 是磁铁类（precision 0.7333）**：行政覆盖会议、场地、物资、活动、后勤，
   主题词面最广，三条跨部门错误全部落进它。

## 五、差距结论（答辩亮点）

| 维度 | 合成集 | 人工集 |
|------|--------|--------|
| accuracy / macro-F1 | 0.9840 / 0.9841 | 0.8898 / 0.8856 |
| 错误率 | 1.60% | 11.02% |
| 错误主因 | 数据标注重叠（75.8%）+ 英文缩写捷径（24.2%） | 越界拒答类缺失 + 部门边界模糊 |
| 错误是否系统性 | 是，99.4% 的错误含 12 个固定主题词中的一个 | 是，集中在 other_chitchat 与 legal_contract |
| 错误是否模型可控 | 否，主题标签在两类间 50:50，期望上限即 50% | 部分可控（拒答类可补），部分受标签体系限制 |
| 修复动作 | 清洗训练数据的主题-标签映射 | 扩标签体系 / 多标签路由 / 拒答机制 |

**为什么合成集分数必然虚高**（可直接写进 PPT）：

1. 训练集与测试集由同一套模板生成，测试集只是训练分布的又一次抽样，属于同源同分布，
   没有考察任何训练时未见过的表达。
2. 合成集上的错误几乎全部来自数据自身的标注重叠与字形捷径，模型把"能拿的分"都拿到了，
   98.4% 更接近"拟合训练分布的上限"，而不是泛化能力。
3. 人工集引入了三类训练集中不存在的现象：越界问句（查他人考勤、CEO 薪酬）、
   部门边界问句（合规 vs 人事、知识产权 vs IT）、词表外术语（"护目镜"在合成语料中出现 0 次）。
4. 结论：**模型真实可用精度应参照人工集的 88.98%，且随人工集继续扩充仍可能下探；
   对外汇报时应同时给出两个数字与差距原因，而不是只报 98.4%。**

## 六、badcase 归类分析

完整分析见 `data/eval/badcase_notes.md`，并由 `src/evaluate.py` 自动并入两份 `badcases.txt` 的第三节，
内容包括：双标签主题的样本级证据（主题词在 train.txt 中的类别分布）、英文缩写捷径的机制、
人工集的四类边界问题、以及 5 条按性价比排序的改进建议。

## 七、复现步骤

```powershell
# 0. 依赖：torch / transformers / scikit-learn / matplotlib / jieba
# 1. 合成测试集
python -m src.evaluate --input data/test.txt --output-dir outputs/eval_test
# 2. 人工评估集
python -m src.evaluate --input data/eval/test_queries.jsonl --output-dir outputs/eval_human
# 3. 生成对比表
python scripts/compare_eval.py --synthetic outputs/eval_test --human outputs/eval_human `
    --out docs/eval_comparison.md
```

每次评估在输出目录产出四件套：`report.txt`（acc / macro-F1 / 分类报告）、
`confusion_matrix.png`、`badcases.txt`（混淆对总览 + 明细 + 人工归因）、
`metrics.json`（机器可读，供对比脚本复用）。

## 八、验收对照

| 验收标准 | 状态 | 位置 |
|----------|------|------|
| report.txt（acc / macro-F1 / 分类报告） | 完成 | `outputs/eval_test/report.txt`、`outputs/eval_human/report.txt` |
| confusion_matrix.png | 完成 | 两个输出目录各一份，9x9 全类别（显式传 labels，避免类别缺席导致矩阵错位） |
| badcases.txt（混淆对归类 + 原因） | 完成 | 两个输出目录各一份，含人工归因分析 |
| 人工评估集扩充到 100+ 条 | 完成 | `data/eval/test_queries.jsonl`，118 条，无重复，覆盖全部 9 类 |
| 合成集 vs 人工集差距结论 | 完成 | 本文第五节 + `docs/eval_comparison.md` |

## 九、过程中发现并修复的问题

1. **权重路径不一致**：`src/config.py` 原默认 `checkpoints/intent_bert/best.pt`，
   本机实际权重在 `models/best.pt`，直接运行会报"未找到模型权重"。
   已新增 `BEST_CKPT = models/best.pt` 并作为 `src/evaluate.py --ckpt` 的默认值。
   注意 `src/predict.py`、`prune.py`、`quantize.py`、`distill.py` 仍指向 `checkpoints/`，
   联调部署前需要统一（属其他成员模块，本次未改动）。
2. **人工集标签越界**：`out_of_scope` 不在 `class.txt`，评估直接报"未知标签名"。
   已归一到第 9 类 `other_chitchat`；同时把 evaluate 的报错信息改成"文件名 + 行号 + 合法标签位置"。
3. **缺本地预训练目录**：`models/` 下只有 `best.pt`，缺 `bert-base-chinese`（分词器与骨干配置）。
   已按 README 的目录约定补齐到 `models/bert-base-chinese/`（`models/` 已在 .gitignore 中，不入库）。
4. **badcases.txt 只有原始清单**：原实现按行输出 `真实\t预测\t文本`，无法回答"哪些类别容易混淆"。
   已重写为"混淆对总览（按错误数降序）-> 明细（高频词证据 + 示例）-> 人工归因"三层结构。
5. **评估不产出机器可读结果**：新增 `metrics.json`，对比脚本因此不需要重跑模型即可复算结论。
6. **混淆矩阵维度不稳**：`confusion_matrix` 未指定 labels，某类在预测中缺席时矩阵会收缩、
   与标签名错位。已显式传 `labels=range(num_labels)`。

## 十、环境说明

本地复现使用的是 `E:\Anaconda\envs\TMF`（Python 3.10.21 + torch 2.14.0+cu126 + transformers 5.16.1，
RTX 3060 Laptop），与 `pyproject.toml` 的 `requires-python >=3.11,<3.12` 不一致：
该 conda 环境已带 CUDA 版 torch，而新建 3.11 环境需要重新下载数 GB 权重。
评估代码只使用 3.10 兼容语法，其余成员按 README 用 `uv venv --python 3.11 .venv` 建环境后可同样复现；
结论只依赖模型权重与数据，不依赖 Python 版本。
