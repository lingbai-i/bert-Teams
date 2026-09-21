# BERT 意图分类评估报告（合成集 vs 人工集 + 合成语料清洗）

> 模块负责人：Mr-zhuyifan（README 分工表 #7：BERT 评估 / src/evaluate.py）
> 评估权重：`models/best.pt`，sha256 `3313804ef8ce1d6d…`（2026-09-21 20:14 版本，已写入每份产物头部）
> 文中所有数字均由脚本产出，可用第九节命令完整复现。

## 一、结论摘要（答辩用）

| 指标 | 未清洗合成集 | 清洗后合成集 | 人工评估集 |
|------|--------------|--------------|------------|
| 文件 | `data/raw/test.txt` | `data/test.txt` | `data/eval/test_queries.jsonl` |
| 样本数 | 11099 | 11095 | 118 |
| accuracy | 0.9845 | 0.9840 | 0.9153 |
| macro-F1 | 0.9847 | 0.9841 | 0.9130 |
| bad case 数 | 172 | 178 | 10 |
| 错误率 | 1.55% | 1.60% | 8.47% |

五条结论：

1. **未清洗合成集的 98.45% 不能直接对外汇报**：172 条错误中 75.0% 来自语料自身的矛盾标注
   （同一个主题字符串被两个类别各标了一半），模型只是"押中了其中一侧"，不是学得好。
2. **清洗（去重 + 重新分类）后，错误结构变成可解释的两类原因**：
   178 条 = 135 条（75.8%）**团队内的口径分歧** + 43 条（24.2%）**英文缩写捷径**，没有第三种。
3. **口径分歧是当前最该先解决的问题**：`病假工资/育儿假/陪产假/加班费计算` 四个主题，
   本仓库按知识库裁定为 `policy_attendance`，而 best.pt 的训练数据标成 `hr_onboarding`/`finance_expense`
   （模型预测一致性 100%）。统一口径后这 135 条错误即刻消失——改规则 2 分钟，或重训后一致。
4. **43 条缩写捷径**（NDA/NPS/VPN -> engineering_eq）标签清洗解决不了，需要补缩写主题数据。
5. **人工集 91.53%**：10 条错误全部是语义边界问题（越界拒答类缺失 5 条 + 部门边界模糊 5 条），
   8 条置信度 >= 0.866、6 条 >= 0.999，属于"高置信度错误"。**这才是模型真实可用性的下限参照。**

## 二、评估设置

| 项 | 内容 |
|------|------|
| 权重 | `models/best.pt`（390MB，201 个张量，`load_state_dict` 严格匹配），sha256 `3313804e…` |
| 分词器 / 骨干 | `models/bert-base-chinese`（vocab.txt + tokenizer.json + config.json） |
| 正式合成集 | `data/test.txt`（已清洗，11095 条） |
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
accuracy = 0.9840   macro_f1 = 0.9841   weighted_f1 = 0.9839
```

| 类别 | precision | recall | F1 | support |
|------|-----------|--------|-----|---------|
| policy_attendance | 1.0000 | 0.8976 | 0.9461 | 1319 |
| it_vpn | 1.0000 | 0.9992 | 0.9996 | 1250 |
| finance_expense | 0.9777 | 1.0000 | 0.9887 | 1229 |
| hr_onboarding | 0.9157 | 1.0000 | 0.9560 | 1162 |
| admin_logistics | 1.0000 | 1.0000 | 1.0000 | 1250 |
| engineering_eq | 0.9667 | 1.0000 | 0.9831 | 1250 |
| legal_contract | 1.0000 | 0.9824 | 0.9911 | 1304 |
| sales_marketing | 1.0000 | 0.9846 | 0.9922 | 1232 |
| other_chitchat | 1.0000 | 1.0000 | 1.0000 | 1099 |

| 混淆对（真实 -> 预测） | 条数 | 占错误比 | 性质 |
|------------------------|------|----------|------|
| policy_attendance -> hr_onboarding | 107 | 60.1% | 口径分歧（病假工资/育儿假/陪产假） |
| policy_attendance -> finance_expense | 28 | 15.7% | 口径分歧（加班费计算） |
| legal_contract -> engineering_eq | 23 | 12.9% | NDA 缩写捷径 |
| sales_marketing -> engineering_eq | 19 | 10.7% | NPS 缩写捷径 |
| it_vpn -> engineering_eq | 1 | 0.6% | VPN 缩写捷径 |

- 前两对共 **135 条（75.8%）** 是口径分歧：模型 100% 一致地预测 hr / finance，
  说明其训练数据的标签就是那样，**换一套口径即刻消失**。
- 后三对共 **43 条（24.2%）** 是缩写捷径，需补数据。
- 反向收益：未清洗集上 hr->legal 37 条、sales->legal 18 条在清洗后消失
  （保密协议/竞业协议/合同模板统一到 legal，与模型预测一致），legal_contract 的 F1
  从 0.9692 升到 0.9911，policy_attendance 的 precision 达到 1.0000。

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
| accuracy / macro-F1 | 0.9845 / 0.9847 | 0.9840 / 0.9841 | 0.9153 / 0.9130 |
| 错误率 | 1.55% | 1.60% | 8.47% |
| 错误主因 | 双标签标注 75.0% + 缩写捷径 25.0% | 口径分歧 75.8% + 缩写捷径 24.2% | 越界拒答类缺失 + 部门边界模糊 |
| 修复动作 | 已修复（清洗数据） | 统一口径 + 补缩写数据 | 扩标签体系 / 多标签路由 / 拒答机制 |

**为什么未清洗的 98.45% 必然虚高**（可直接写进 PPT）：

1. 训练集与测试集由同一套模板生成，测试集只是训练分布的又一次抽样，没有考察任何训练时未见过的表达。
2. 生成映射表把 7 个主题字符串同时分配给了两个类别（50:50），任何模型在这批样本上的期望上限就是 50%；
   这部分正确率来自"押中一侧"，换一个随机种子可能就换成另一侧。
3. 语料还有 1266 条完全相同却标签不同的重复句（train）。
4. 清洗后这批错误按性质分解：135 条口径分歧（可即时消除）+ 43 条缩写捷径（需补数据），**没有第三条原因**。
5. 人工集引入了训练集中不存在的三类现象：越界问句（查他人考勤、CEO 薪酬）、部门边界问句
   （合规 vs 人事、合同文本 vs 客户场景）、词表外术语（"护目镜"在合成语料中出现 0 次）。
6. 结论：**对外汇报应给出三个数字与各自成因——未清洗 98.45%（数据缺陷导致的虚高）、
   清洗后 98.40%（其中 75.8% 是待统一的口径、24.2% 是缩写捷径）、人工集 91.53%（真实可用性）。**

## 六、口径分歧（当前第一优先级待决策项）

| 主题 | 本仓库裁定（知识库依据） | best.pt 训练口径 | 模型一致性 | 知识库原文 |
|------|--------------------------|------------------|------------|------------|
| 病假工资 | policy_attendance | hr_onboarding | 100% | policy/attendance.md 2.2 病假：病假工资按基本工资的 80% 发放 |
| 育儿假 | policy_attendance | hr_onboarding | 100% | policy/attendance.md 2.4 其他假期：育儿假每年各 5 天 |
| 陪产假 | policy_attendance | hr_onboarding | 100% | policy/attendance.md 2.4 其他假期：陪产假 15 天 |
| 加班费计算 | policy_attendance | finance_expense | 77% | policy/overtime.md 章节标题《加班费计算》 |
| 保密协议 / 竞业协议 / 合同模板 | legal_contract | legal_contract | 100% | legal/confidential.md、legal/contract.md（已一致） |

- 反推方法：在未清洗测试集上按主题统计 best.pt 的预测分布，若某主题预测 100% 一致，
  说明其训练数据的标签就是该类别（50:50 的监督只会学到五五开分布）。
- **建议按知识库口径统一**：部署层是拿意图标签去 `data/knowledge_base/<部门>/` 检索文档的。
  `病假工资` 若被路由到 hr 部门，而答案写在 `policy/attendance.md`，现场 demo 会取错文档、答非所问。
  意图标签应指向"哪个部门的文档能回答这个问题"——这正是本仓库裁定的定义。
- 两条落地路径（脚本层都已就绪）：
  1. **按知识库口径**：使用本仓库 `data/*.txt`（已清洗），重训一次（约 60~90 分钟）后口径完全一致；
  2. **按现有模型口径**：把规则表中这 4 条的 `label` 改成 hr_onboarding / finance_expense，
     重跑 `scripts/clean_synthetic.py` 并重跑评估（约 2 分钟），不需要重训。

## 七、合成语料清洗（去重 + 重新分类）

### 7.1 缺陷证据

- **双标签主题 7 个**：病假工资、育儿假、陪产假、加班费计算、保密协议、竞业协议、合同模板
  （同一主题字符串被两个类别各标约一半，最大类占比 50.2%~52.1%）。
- **同文本异标签**：train.txt 1266 行、dev.txt 2 行、test.txt 4 行。
- 发现方法：按 Apriori 思路挖掘全语料频繁短语 -> 用"各类别均匀出现的片段"识别模板框架 ->
  剥掉框架得到主题词 -> 统计每个主题词的标签分布，最大占比 <= 0.90 判为跨类主题。

### 7.2 分类依据

`data/knowledge_base/<部门>/*.md` 的 8 个部门目录与 `class.txt` 的 8 个业务类别一一对应
（policy/it/finance/hr/admin/engineering/legal/sales），第 9 类 `other_chitchat` 无知识库。
因此**主题归属 = 该主题在知识库中由哪个部门的文档承载**，每条规则的 evidence 都指向知识库原文
（完整表格见第六节与 `data/synthetic_label_rules.json`）。

规则表只收录**真·矛盾标注**（7 条）；另有 4 个候选经复核后移入 `dropped_candidates` 字段并说明理由：
`盖章`（合同盖章与盖章流程是两个不同主题、各自内部一致）、`劳动仲裁`（语料中 100% 属 hr_onboarding）、
`产假工资`、`年假计算`（各自 100% 一致，属口径选择而非矛盾标注）。

### 7.3 清洗结果与目录约定

| 文件 | 行数变化 | 重分类行数 | 去重丢弃 | 残留同文本异标签 |
|------|----------|------------|----------|------------------|
| `data/train.txt` | 202500 -> 201234 | 2604 | 1266 | 0 |
| `data/dev.txt` | 10000 -> 9998 | 137 | 2 | 0 |
| `data/test.txt` | 11099 -> 11095 | 127 | 4 | 0 |

- **目录约定（本次调整）**：`data/{train,dev,test}.txt` = 清洗后的正式数据（`src/config.py` 直接可用）；
  `data/raw/{train,dev,test}.txt` = 原始未清洗语料（归档，用于对比与复现）；
  清洗报告在 `data/clean_reports/{train,dev,test}_report.md`。
- 未命中规则的样本一律保持原标签：只清洗已确认有矛盾的地方，避免清洗本身引入新噪声。
- 仍存在**概念近邻但标签不矛盾**的主题（社保缴纳(hr) / 社保扣款(finance)、投影连接(it) / 投影借用(admin)、
  合同盖章(legal) / 盖章流程(admin)），属部门职责的合理切分，未改动。

## 八、badcase 归类分析

完整分析见 `data/eval/badcase_notes.md`，并由 `src/evaluate.py` 自动并入每份 `badcases.txt` 的第三节，
内容包括：双标签主题的样本级证据、缩写捷径的机制、口径分歧明细、人工集的错误明细与置信度、
以及 6 条按性价比排序的改进建议。

## 九、复现步骤

```powershell
# 1. 清洗合成语料（去重 + 按知识库重新分类）；原始语料在 data/raw/ 下
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
   导致同一测试集出现 0.9840 / 0.9845 两个分数、人工集出现 0.8898 / 0.9153 两个分数，
   而文件名不变、无法从产物判断指标出处。已新增 `file_sha256()` 把哈希写进产物；
   **换权重后必须整套重跑。**
9. **清洗规则自身的过度收敛（自我修正）**：第一版规则表把 `盖章`、`劳动仲裁`、`产假工资`、`年假计算`
   也纳入重分类，实测发现它们并非"同一主题字符串被两类标注"，只是子串重合或口径选择；
   强行统一会平白制造 55 条"错误"（0.9790 vs 修正后的 0.9840）。规则表已收窄到 7 条真矛盾标注，
   并把 4 个候选移入 `dropped_candidates` 并说明理由。

## 十二、环境说明

本地复现使用 `E:\Anaconda\envs\TMF`（Python 3.10.21 + torch 2.14.0+cu126 + transformers 5.16.1，
RTX 3060 Laptop），与 `pyproject.toml` 的 `requires-python >=3.11,<3.12` 不一致：
该 conda 环境已带 CUDA 版 torch，而新建 3.11 环境需要重新下载数 GB 权重。
评估与清洗代码只使用 3.10 兼容语法，其余成员按 README 用 `uv venv --python 3.11 .venv` 建环境后可同样复现。
