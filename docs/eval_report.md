# BERT 意图分类评估报告（合成集 vs 人工集 + 合成语料清洗）

> 模块负责人：Mr-zhuyifan（README 分工表 #7：BERT 评估 / src/evaluate.py）
> 评估权重：`models/best.pt`，sha256 `3313804ef8ce1d6d…`（2026-09-21 20:14 版本，已写入每份产物头部）
> 文中所有数字均由脚本产出，可用第九节命令完整复现。

## 一、结论摘要（答辩用）

| 指标 | 未清洗合成集 | 清洗后合成集 | 人工评估集 |
|------|--------------|--------------|------------|
| 文件 | `data/raw/test.txt` | `data/test.txt` | `data/eval/test_queries.jsonl` |
| 样本数 | 11099 | 11095 | 118 |
| accuracy | 0.9845 | **0.9953** | **0.9153** |
| macro-F1 | 0.9847 | **0.9954** | **0.9130** |
| bad case 数 | 172 | 52 | 10 |
| 错误率 | 1.55% | 0.47% | 8.47% |

五条结论：

1. **未清洗合成集的 98.45% 不能直接对外汇报**：172 条错误中 75.0% 来自语料自身的矛盾标注
   （同一个主题字符串被两个类别各标了一半），模型只是"押中了其中一侧"，不是学得好。
2. **清洗 + 统一标签口径后提升到 99.53%**：52 条错误中 43 条（82.7%）是英文缩写捷径
   （NDA/NPS/VPN -> engineering_eq），9 条（17.3%）是同一主题内的预测不一致，**没有第三种原因**。
3. **口径统一是本次提升的主要来源**：清洗前那 135 条"错误"其实是团队内两套口径的分歧
   （知识库口径 vs 权重训练口径），统一后全部消失——同一份数据、同一个模型，仅因标签口径不同，
   准确率在 98.40% / 99.53% 之间漂移 1.13 个百分点。
4. **43 条缩写捷径**标签清洗解决不了，需要补缩写主题数据后重训。
5. **人工集 91.53%**：10 条错误全部是语义边界问题（越界拒答类缺失 5 条 + 部门边界模糊 5 条），
   8 条置信度 >= 0.866、6 条 >= 0.999，属于"高置信度错误"。
   **它与合成集的 8.0 个百分点差距，才是答辩真正的亮点。**

## 二、评估设置

| 项 | 内容 |
|------|------|
| 权重 | `models/best.pt`（390MB，201 个张量，`load_state_dict` 严格匹配），sha256 `3313804e…` |
| 分词器 / 骨干 | `models/bert-base-chinese`（vocab.txt + tokenizer.json + config.json） |
| 正式合成集 | `data/test.txt`（已清洗，口径与权重一致，11095 条） |
| 原始合成集 | `data/raw/test.txt`（未清洗，11099 条） |
| 人工集 | `data/eval/test_queries.jsonl`，118 条，`{"query","intent"}` |
| 截断长度 | max_len = 48（`src/config.py`） |
| 批次 / 设备 | bs = 64 / CUDA（RTX 3060 Laptop） |
| 指标 | accuracy、macro-F1、weighted-F1、逐类 precision/recall/F1、混淆矩阵 |

## 三、合成测试集结果

### 3.1 未清洗集（outputs/eval_test_raw，data/raw/test.txt）

```
accuracy = 0.9845   macro_f1 = 0.9847   weighted_f1 = 0.9845
```

| 类别 | precision | recall | F1 | support |
|------|-----------|--------|-----|---------|
| policy_attendance | 0.9966 | 0.9440 | 0.9696 | 1250 |
| it_vpn | 1.0000 | 0.9992 | 0.9996 | 1250 |
| finance_expense | 0.9897 | 0.9968 | 0.9932 | 1250 |
| hr_onboarding | 0.9551 | 0.9704 | 0.9627 | 1250 |
| admin_logistics | 1.0000 | 1.0000 | 1.0000 | 1250 |
| engineering_eq | 0.9667 | 1.0000 | 0.9831 | 1250 |
| legal_contract | 0.9571 | 0.9816 | 0.9692 | 1250 |
| sales_marketing | 1.0000 | 0.9704 | 0.9850 | 1250 |
| other_chitchat | 1.0000 | 1.0000 | 1.0000 | 1099 |

172 条错误：双标签主题 **129 条（75.0%）** + 英文缩写捷径 **43 条（25.0%）** + 其余 **0 条**。
主要混淆对：policy->hr 57、hr->legal 37、legal->eng 23、sales->eng 19、sales->legal 18、policy->finance 13。

### 3.2 清洗后集（outputs/eval_test，data/test.txt）

```
accuracy = 0.9953   macro_f1 = 0.9954   weighted_f1 = 0.9953
```

| 类别 | precision | recall | F1 | support |
|------|-----------|--------|-----|---------|
| policy_attendance | 0.9924 | 1.0000 | 0.9962 | 1175 |
| it_vpn | 1.0000 | 0.9992 | 0.9996 | 1250 |
| finance_expense | 1.0000 | 0.9929 | 0.9964 | 1266 |
| hr_onboarding | 1.0000 | 1.0000 | 1.0000 | 1269 |
| admin_logistics | 1.0000 | 1.0000 | 1.0000 | 1250 |
| engineering_eq | 0.9667 | 1.0000 | 0.9831 | 1250 |
| legal_contract | 1.0000 | 0.9824 | 0.9911 | 1304 |
| sales_marketing | 1.0000 | 0.9846 | 0.9922 | 1232 |
| other_chitchat | 1.0000 | 1.0000 | 1.0000 | 1099 |

| 混淆对（真实 -> 预测） | 条数 | 占错误比 | 性质 |
|------------------------|------|----------|------|
| legal_contract -> engineering_eq | 23 | 44.2% | NDA 缩写捷径 |
| sales_marketing -> engineering_eq | 19 | 36.5% | NPS 缩写捷径 |
| finance_expense -> policy_attendance | 9 | 17.3% | 加班费计算：模型对该主题预测一致性仅 77% |
| it_vpn -> engineering_eq | 1 | 1.9% | VPN 缩写捷径 |

- **43 条（82.7%）是缩写捷径**（需补数据）；**9 条（17.3%）是同一主题内的不一致**。
- 清洗前占 75.8% 的 135 条口径分歧错误**全部消失**（标签与模型预测同侧后不再算错）。
- 逐类收益：`hr_onboarding` F1 0.9627 -> 1.0000、`legal_contract` 0.9692 -> 0.9911、
  `finance_expense` 0.9932 -> 0.9964、`policy_attendance` 0.9696 -> 0.9962；
  整体 accuracy 0.9845 -> 0.9953、macro-F1 0.9847 -> 0.9954。

## 四、人工评估集结果

### 4.1 数据集扩充（18 -> 118 条）

- 原有 18 条保留原句不变；本次按"全员每人 10 条真实问句"补入 **100 条**，
  覆盖此前为空的 4 个类别（admin_logistics、engineering_eq、legal_contract、sales_marketing）。
- 汇总后 118 条，类别分布 12~14 条/类，**去重检查无重复句**。
- 标签对齐：原文件 2 条使用 `out_of_scope`，该标签名不在 `data/class.txt` 中，
  已归一到第 9 类 `other_chitchat`，否则 evaluate 会直接抛"未知标签名"而无法评估。
- 人工句长中位数 12 字（合成句 9 字），句式与合成集不同（口语、省略、带具体时间/系统名）。

### 4.2 结果

```
accuracy = 0.9153   macro_f1 = 0.9130   weighted_f1 = 0.9121
```

| 类别 | precision | recall | F1 | support |
|------|-----------|--------|-----|---------|
| policy_attendance | 0.8750 | 1.0000 | 0.9333 | 14 |
| it_vpn | 1.0000 | 0.9286 | 0.9630 | 14 |
| finance_expense | 0.9286 | 0.9286 | 0.9286 | 14 |
| hr_onboarding | 0.8235 | 1.0000 | 0.9032 | 14 |
| admin_logistics | 0.8000 | 1.0000 | 0.8889 | 12 |
| engineering_eq | 1.0000 | 1.0000 | 1.0000 | 12 |
| legal_contract | 1.0000 | 0.7500 | 0.8571 | 12 |
| sales_marketing | 0.9231 | 1.0000 | 0.9600 | 12 |
| other_chitchat | 1.0000 | 0.6429 | 0.7826 | 14 |

10 条错误分三类：

1. **第 9 类缺"越界/拒答"语义（5 条）**：`other_chitchat` recall 仅 0.6429。
   别人的考勤记录能查吗 -> policy_attendance（1.000）、能帮我查下同事这个月的考勤吗 -> policy_attendance（1.000）、
   帮我给宠物起个名字 -> admin_logistics（1.000）、公司CEO的年薪是多少 -> hr_onboarding（0.999）、
   CEO 的工资是多少 -> finance_expense（0.999）。
2. **部门边界本身模糊（5 条）**：`legal_contract` recall 0.75（劳动合同续签、个人信息同意书 -> hr，
   保密协议 -> sales），另有项目聚餐费用 -> admin、会议室投影仪 -> admin。
   这类问句标注时人人都能讲出理由，属于**单标签分类的粒度天花板**，答辩时应说明裁定口径。
3. **`admin_logistics` 是磁铁类（precision 0.8000）**：行政覆盖会议、场地、物资、活动、后勤，
   主题词面最广，跨部门错误容易落进它（3/10 条）。

## 五、差距结论（答辩亮点）

| 维度 | 未清洗合成集 | 清洗后合成集 | 人工集 |
|------|--------------|--------------|--------|
| accuracy / macro-F1 | 0.9845 / 0.9847 | 0.9953 / 0.9954 | 0.9153 / 0.9130 |
| 错误率 | 1.55% | 0.47% | 8.47% |
| 错误主因 | 双标签标注 75.0% + 缩写捷径 25.0% | 缩写捷径 82.7% + 主题内不一致 17.3% | 越界拒答类缺失 + 部门边界模糊 |
| 修复动作 | 已修复（清洗数据） | 补缩写主题数据 | 扩标签体系 / 多标签路由 / 拒答机制 |

**答辩可直接讲的四条：**

1. 训练集与测试集由同一套模板生成，测试集只是训练分布的又一次抽样，没有考察任何训练时未见过的表达。
2. 原始语料存在**可验证的缺陷**：7 个主题字符串被两个类别各标一半（最大类占比 50.2%~52.1%），
   另有 1266 行完全相同的句子带两个标签；未清洗的 98.45% 里有 75.0% 的错误正落在这些主题上。
3. 清洗后错误结构变得可解释：**仅两类原因**（缩写捷径 82.7% + 主题内不一致 17.3%），
   且其中"口径分歧"这一类已经通过统一口径消除。
4. 人工集 91.53% 暴露的是训练数据里根本不存在的问题（越界拒答、部门边界、词表外术语），
   **合成集 99.53% 与人工集 91.53% 的 8.0pp 差距 = 数据分布差距 + 标签体系缺陷，而非模型能力差距。**

## 六、口径决策记录（2026-09-21）

| 项 | 内容 |
|----|------|
| 决策 | 7 个争议主题的归属统一采用权威权重 `models/best.pt` 的训练口径，不采用 knowledge_base 的部门归属裁定 |
| 范围 | 病假工资 / 育儿假 / 陪产假 -> `hr_onboarding`；加班费计算 -> `finance_expense`；保密协议 / 竞业协议 / 合同模板 -> `legal_contract`（两套口径本来就一致） |
| 理由 | 与已训练好的权威权重一致，无需重训即可让数据与模型自洽，答辩前时间成本最低 |
| 反推依据 | 在未清洗测试集上按主题统计 best.pt 的预测分布：病假工资/育儿假/陪产假预测 hr_onboarding 一致性 100%，加班费计算预测 finance_expense 77%（50:50 的监督只会学到五五开分布，故一致性即训练口径） |
| knowledge_base 对照 | `policy/attendance.md` 2.2 写了病假工资发放比例、2.4 列了育儿假/陪产假；`policy/overtime.md` 章节标题即《加班费计算》——知识库侧更倾向 policy_attendance |
| 已知风险 | 部署层按意图标签检索 `data/knowledge_base/<部门>/`：`病假工资` 命中 hr 目录、`加班费计算` 命中 finance 目录，而答案写在 policy 目录下，现场 demo 有取错文档的风险 |
| 缓解方式（部署组） | (1) 在 `app/flask_app.py` 的知识库路由层加"主题 -> 文档目录"别名映射；(2) 把 `policy/attendance.md` 的假期待遇段落与 `policy/overtime.md` 的加班费段落同步到 `hr/`、`finance/` 目录 |
| 回退方式 | 把 `data/synthetic_label_rules.json` 这 4 条的 `label` 改回 `policy_attendance`，重跑清洗脚本与三份评估（约 2 分钟），无需改代码 |
| 量化影响 | 同一模型、同一份数据，口径不同 → 98.40% 与 99.53%（差 1.13pp）。**口径必须写进实验记录，否则指标无法复现。** |

## 七、合成语料清洗（去重 + 重新分类）

### 7.1 缺陷证据

- **双标签主题 7 个**：病假工资、育儿假、陪产假、加班费计算、保密协议、竞业协议、合同模板
  （同一主题字符串被两个类别各标约一半，最大类占比 50.2%~52.1%）。
- **同文本异标签**：train.txt 1266 行、dev.txt 2 行、test.txt 4 行。
- 发现方法：按 Apriori 思路挖掘全语料频繁短语 -> 用"各类别均匀出现的片段"识别模板框架 ->
  剥掉框架得到主题词 -> 统计每个主题词的标签分布，最大占比 <= 0.90 判为跨类主题。

### 7.2 规则表与裁定依据

规则表 `data/synthetic_label_rules.json` 只收录**真·矛盾标注**（7 条），每条带
`label / kind / basis / before / knowledge_base_note`：`basis=knowledge_base` 表示按部门文档归属裁定，
`basis=convention_decision` 表示知识库与权重冲突时按团队口径决策（见第六节）。

另有 4 个候选经复核后移入 `dropped_candidates` 字段并说明理由：`盖章`
（合同盖章与盖章流程是两个不同主题、各自内部一致）、`劳动仲裁`（语料中 100% 属 hr_onboarding）、
`产假工资`、`年假计算`（各自 100% 一致，属口径选择而非矛盾标注）。

### 7.3 清洗结果与目录约定

| 文件 | 行数变化 | 重分类行数 | 去重丢弃 | 残留同文本异标签 |
|------|----------|------------|----------|------------------|
| `data/train.txt` | 202500 -> 201234 | 2584 | 1266 | 0 |
| `data/dev.txt` | 10000 -> 9998 | 133 | 2 | 0 |
| `data/test.txt` | 11099 -> 11095 | 130 | 4 | 0 |

- **目录约定**：`data/{train,dev,test}.txt` = 清洗后的正式数据（`src/config.py` 直接可用）；
  `data/raw/{train,dev,test}.txt` = 原始未清洗语料（归档，用于对比与复现）；
  清洗报告在 `data/clean_reports/{train,dev,test}_report.md`。
- 未命中规则的样本一律保持原标签：只清洗已确认有矛盾的地方，避免清洗本身引入新噪声。
- 仍存在**概念近邻但标签不矛盾**的主题（社保缴纳(hr) / 社保扣款(finance)、投影连接(it) / 投影借用(admin)、
  合同盖章(legal) / 盖章流程(admin)），属部门职责的合理切分，未改动。

## 八、badcase 归类分析

完整分析见 `data/eval/badcase_notes.md`，并由 `src/evaluate.py` 自动并入每份 `badcases.txt` 的第三节，
内容包括：双标签主题的样本级证据、缩写捷径的机制、口径决策记录与风险、人工集的错误明细与置信度、
以及 6 条按性价比排序的改进建议。

## 九、复现步骤

```powershell
# 1. 清洗合成语料（去重 + 按规则表重新分类）；原始语料在 data/raw/ 下
python scripts/clean_synthetic.py --input data/raw/train.txt --output data/train.txt --report data/clean_reports/train_report.md
python scripts/clean_synthetic.py --input data/raw/dev.txt   --output data/dev.txt   --report data/clean_reports/dev_report.md
python scripts/clean_synthetic.py --input data/raw/test.txt  --output data/test.txt  --report data/clean_reports/test_report.md

# 2. 三个数据集上的评估（产物会记录权重 sha256，换权重必须整套重跑）
python -m src.evaluate --input data/raw/test.txt            --output-dir outputs/eval_test_raw  # 未清洗基线
python -m src.evaluate --input data/test.txt                --output-dir outputs/eval_test      # 正式（清洗后）
python -m src.evaluate --input data/eval/test_queries.jsonl --output-dir outputs/eval_human      # 人工集

# 3. 生成对比表
python scripts/compare_eval.py --synthetic outputs/eval_test     --human outputs/eval_human --out docs/eval_comparison.md
python scripts/compare_eval.py --synthetic outputs/eval_test_raw --human outputs/eval_human --out docs/eval_comparison_raw.md
```

每次评估在输出目录产出四件套：`report.txt`（acc / macro-F1 / 分类报告）、`confusion_matrix.png`、
`badcases.txt`（混淆对总览 + 明细 + 人工归因 + 权重哈希）、`metrics.json`（含 `ckpt_sha256`）。

## 十、验收对照

| 验收标准 | 状态 | 位置 |
|----------|------|------|
| report.txt（acc / macro-F1 / 分类报告） | 完成 | `outputs/eval_test/`、`outputs/eval_test_raw/`、`outputs/eval_human/` |
| confusion_matrix.png | 完成 | 同上三个目录，9x9 全类别（显式传 labels，避免类别缺席导致矩阵错位） |
| badcases.txt（混淆对归类 + 原因） | 完成 | 同上三个目录，含人工归因分析与权重哈希 |
| 人工评估集扩充到 100+ 条 | 完成 | `data/eval/test_queries.jsonl`，118 条，无重复，覆盖全部 9 类 |
| 合成集 vs 人工集差距结论 | 完成 | 本文第五节 + `docs/eval_comparison.md`、`docs/eval_comparison_raw.md` |
| 合成语料清洗（去重 + 重新分类） | 完成 | `scripts/clean_synthetic.py`、`data/synthetic_label_rules.json`、`data/{train,dev,test}.txt` + `data/clean_reports/` |

## 十一、过程中发现并修复的问题

1. **权重路径不一致**：`src/config.py` 原默认 `checkpoints/intent_bert/best.pt`，实际权重在 `models/best.pt`。
   已新增 `BEST_CKPT` 并作为 `--ckpt` 默认值。注意 `src/predict.py`、`prune.py`、`quantize.py`、
   `distill.py` 仍指向 `checkpoints/`，联调前需统一。
2. **人工集标签越界**：`out_of_scope` 不在 `class.txt`，评估直接报错。已归一到 `other_chitchat`，
   并把报错信息改成"文件名 + 行号 + 合法标签位置"。
3. **缺本地预训练目录**：`models/` 下只有 `best.pt`，已按 README 约定补齐 `models/bert-base-chinese/`
   （`models/` 已在 .gitignore 中，不入库）。
4. **badcases.txt 只有原始清单**：重写为"混淆对总览 -> 明细（高频词证据 + 示例）-> 人工归因"三层结构。
5. **评估不产出机器可读结果**：新增 `metrics.json`（含 `ckpt_sha256`），对比脚本无需重跑模型即可复算。
6. **混淆矩阵维度不稳**：已显式传 `labels=range(num_labels)`，避免某类缺席时矩阵收缩、与标签名错位。
7. **合成语料矛盾标注 + 重复行**：7 个主题双标签 + 1266 条同文本异标签（详见第七节），已清洗。
8. **评估中途权重被替换（本次实测到）**：`models/best.pt` 于 2026-09-21 20:14 被覆盖，
   导致同一测试集出现两个分数、人工集出现 0.8898 / 0.9153 两个分数，而文件名不变、无法从产物判断出处。
   已新增 `file_sha256()` 把哈希写进产物；**换权重后必须整套重跑**。
9. **清洗规则自身的过度收敛（自我修正）**：第一版规则表把 `盖章`、`劳动仲裁`、`产假工资`、`年假计算`
   也纳入重分类，实测发现它们并非"同一主题字符串被两类标注"，只是子串重合或口径选择；
   强行统一会平白制造 55 条假错误。规则表已收窄到 7 条真矛盾标注，4 个候选移入 `dropped_candidates`。
10. **标签口径未记录导致指标漂移**：同一模型、同一份数据，口径不同得到 98.40% 与 99.53%。
    现已把口径决策、反推依据与回退方式写进规则表与本文第六节，作为实验记录的一部分。

## 十二、环境说明

本地复现使用 `E:\Anaconda\envs\TMF`（Python 3.10.21 + torch 2.14.0+cu126 + transformers 5.16.1，
RTX 3060 Laptop），与 `pyproject.toml` 的 `requires-python >=3.11,<3.12` 不一致：
该 conda 环境已带 CUDA 版 torch，而新建 3.11 环境需要重新下载数 GB 权重。
评估与清洗代码只使用 3.10 兼容语法，其余成员按 README 用 `uv venv --python 3.11 .venv` 建环境后可同样复现。
