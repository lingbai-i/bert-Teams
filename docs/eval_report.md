# BERT 意图分类评估报告（合成集 vs 人工集 + 合成语料清洗）

> 模块负责人：Mr-zhuyifan（README 分工表 #7：BERT 评估 / src/evaluate.py）
> 评估权重：`models/best.pt`，sha256 `3313804ef8ce1d6d…`（2026-09-21 20:14 版本，已写入每份产物头部）
> 文中所有数字均由脚本产出，可用第八节命令完整复现。

## 一、结论摘要（答辩用）

| 指标 | 未清洗合成集 | 清洗后合成集 | 人工评估集 |
|------|--------------|--------------|------------|
| 样本数 | 11099 | 11095 | 118 |
| accuracy | 0.9845 | 0.9790 | 0.9153 |
| macro-F1 | 0.9847 | 0.9792 | 0.9130 |
| bad case 数 | 172 | 233 | 10 |
| 错误率 | 1.55% | 2.10% | 8.47% |

五条结论：

1. **未清洗合成集的 98.45% 不能直接对外汇报**：172 条错误中 75.0% 来自语料自身的矛盾标注
   （同一个主题字符串被两个类别各标了一半），模型只是"押中了其中一侧"，不是学得好。
2. **清洗（去重 + 按知识库重新分类）后，同一个模型的准确率反而降到 97.90%**
   （错误 172 -> 233）：因为它是在脏数据上训练的，学到的是错误归属。多出的 61 条错误里
   81.5% 属于"重训即可消除"，这恰好是清洗生效的证据。
3. **剩下 43 条（18.5%）是英文缩写捷径**（NDA/NPS/VPN -> engineering_eq），标签清洗无法解决，
   需要补缩写主题数据。
4. **人工集 91.53%**：10 条错误全部是语义边界问题（越界拒答类缺失 5 条 + 部门边界模糊 5 条），
   其中 8 条置信度 >= 0.866、6 条 >= 0.999，属于"高置信度错误"。
5. **差距结论**：清洗后合成集 97.90% vs 人工集 91.53%，差 6.37pp（未清洗时为 6.92pp）。
   差距略微收窄，更重要的是分数变得**可解释**——每个错误都能说清是数据缺陷还是标签体系缺陷。

## 二、评估设置

| 项 | 内容 |
|------|------|
| 权重 | `models/best.pt`（390MB，201 个张量，`load_state_dict` 严格匹配，missing/unexpected 均为 0），sha256 `3313804e…` |
| 分词器 / 骨干 | `models/bert-base-chinese`（vocab.txt + tokenizer.json + config.json） |
| 合成集 | `data/test.txt`（未清洗）与 `data/clean/test.txt`（清洗后），`文本\t标签ID` |
| 人工集 | `data/eval/test_queries.jsonl`，118 条，`{"query","intent"}` |
| 截断长度 | max_len = 48（`src/config.py`） |
| 批次 / 设备 | bs = 64 / CUDA（RTX 3060 Laptop） |
| 指标 | accuracy、macro-F1、weighted-F1、逐类 precision/recall/F1、混淆矩阵 |

## 三、合成测试集结果

### 3.1 未清洗集（outputs/eval_test）

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

172 条错误的构成与归因：

| 混淆对（真实 -> 预测） | 条数 | 归因 |
|------------------------|------|------|
| policy_attendance -> hr_onboarding | 57 | 病假工资/育儿假/陪产假在两类间 50:50 标注 |
| hr_onboarding -> legal_contract | 37 | 保密协议/竞业协议在两类间 50:50 标注 |
| legal_contract -> engineering_eq | 23 | NDA 触发"英文字母串 -> engineering_eq"捷径 |
| sales_marketing -> engineering_eq | 19 | NPS 同上 |
| sales_marketing -> legal_contract | 18 | 合同模板在两类间 51:49 标注 |
| policy_attendance -> finance_expense | 13 | 加班费计算在两类间 50:50 标注 |
| finance_expense -> policy_attendance | 4 | 同上 |
| it_vpn -> engineering_eq | 1 | VPN 缩写捷径 |

按成因拆分：**清洗涉及的主题 129 条（75.0%）+ 英文缩写捷径 43 条（25.0%）+ 其余 0 条**，
没有第三条原因。

### 3.2 清洗后集（outputs/eval_test_clean）

```
accuracy = 0.9790   macro_f1 = 0.9792   weighted_f1 = 0.9789
```

| 类别 | precision | recall | F1 | support |
|------|-----------|--------|-----|---------|
| policy_attendance | 1.0000 | 0.8770 | 0.9345 | 1350 |
| it_vpn | 1.0000 | 0.9992 | 0.9996 | 1250 |
| finance_expense | 0.9777 | 1.0000 | 0.9887 | 1229 |
| hr_onboarding | 0.8913 | 1.0000 | 0.9425 | 1131 |
| admin_logistics | 0.9808 | 1.0000 | 0.9903 | 1226 |
| engineering_eq | 0.9667 | 1.0000 | 0.9831 | 1250 |
| legal_contract | 1.0000 | 0.9646 | 0.9820 | 1328 |
| sales_marketing | 1.0000 | 0.9846 | 0.9922 | 1232 |
| other_chitchat | 1.0000 | 1.0000 | 1.0000 | 1099 |

| 混淆对（真实 -> 预测） | 条数 | 占错误比 | 性质 |
|------------------------|------|----------|------|
| policy_attendance -> hr_onboarding | 138 | 59.2% | 旧模型学到 hr 一侧（病假工资/育儿假/陪产假/产假工资） |
| policy_attendance -> finance_expense | 28 | 12.0% | 旧模型学到 finance 一侧（加班费计算） |
| legal_contract -> admin_logistics | 24 | 10.3% | 旧模型学到 admin 一侧（盖章流程） |
| legal_contract -> engineering_eq | 23 | 9.9% | NDA 缩写捷径 |
| sales_marketing -> engineering_eq | 19 | 8.2% | NPS 缩写捷径 |
| it_vpn -> engineering_eq | 1 | 0.4% | VPN 缩写捷径 |

- 前 3 对共 **190 条（81.5%）** 是"模型学错了归属"，用 `data/clean/train.txt` 重训即可消除。
- 后 3 对共 **43 条（18.5%）** 是缩写捷径，需要补数据。
- 反向收益：未清洗集上 hr_onboarding -> legal_contract 的 37 条混淆在清洗后消失
  （标签统一到 legal 一侧，与模型预测一致）。

## 四、人工评估集结果

### 4.1 数据集扩充（18 -> 118 条）

- 原有 18 条保留原句不变；本次按"全员每人 10 条真实问句"补入 **100 条**，
  覆盖此前为空的 4 个类别（admin_logistics、engineering_eq、legal_contract、sales_marketing）。
- 汇总后 118 条，类别分布 12~14 条/类，**去重检查无重复句**。
- 标签对齐：原文件 2 条使用 `out_of_scope`，该标签名不在 `data/class.txt` 中，
  已归一到第 9 类 `other_chitchat`，否则 evaluate 会直接抛"未知标签名"而无法评估。
- 问句写法贴合真实员工口径（口语、省略、带具体时间/系统名），与合成集的"主题词 + 模板后缀"句式不同：
  人工句长中位数 12 字，合成句 9 字。

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

1. **第 9 类缺"越界/拒答"语义（5 条）**：`other_chitchat` recall 仅 0.6429。训练语料中该类全是纯闲聊，
   句子一沾业务词就被拉走：别人的考勤记录能查吗 -> policy_attendance（1.000）、
   能帮我查下同事这个月的考勤吗 -> policy_attendance（1.000）、帮我给宠物起个名字 -> admin_logistics（1.000）、
   公司CEO的年薪是多少 -> hr_onboarding（0.999）、CEO 的工资是多少 -> finance_expense（0.999）。
2. **部门边界本身模糊（5 条）**：`legal_contract` recall 0.75（劳动合同续签、个人信息同意书 -> hr，
   保密协议 -> sales），另有项目聚餐的费用走哪个科目（finance -> admin）、
   会议室的投影仪连不上我的笔记本（it -> admin）。这类问句标注时人人都能讲出理由，
   属于**单标签分类的粒度天花板**，答辩时应说明裁定口径。
3. **`admin_logistics` 是磁铁类（precision 0.8000）**：行政覆盖会议、场地、物资、活动、后勤，
   主题词面最广，跨部门错误容易落进它（3/10 条）。

## 五、差距结论（答辩亮点）

| 维度 | 未清洗合成集 | 清洗后合成集（旧模型） | 人工集 |
|------|--------------|------------------------|--------|
| accuracy / macro-F1 | 0.9845 / 0.9847 | 0.9790 / 0.9792 | 0.9153 / 0.9130 |
| 错误率 | 1.55% | 2.10% | 8.47% |
| 错误主因 | 双标签标注 75.0% + 缩写捷径 25.0% | 旧模型学错归属 81.5% + 缩写捷径 18.5% | 越界拒答类缺失 + 部门边界模糊 |
| 修复路径 | 已修复（清洗数据） | 重训即可消除 190/233 条 | 扩标签体系 / 多标签路由 / 拒答机制 |

**为什么未清洗的 98.45% 必然虚高**（可直接写进 PPT）：

1. 训练集与测试集由同一套模板生成，测试集只是训练分布的又一次抽样，没有考察任何训练时未见过的表达。
2. 生成映射表把 7 个主题字符串同时分配给了两个类别（50:50），任何模型在这批样本上的期望上限就是 50%；
   这部分正确率来自"押中一侧"，换一个随机种子可能就换成另一侧。
3. 语料还有 1266 条完全相同却标签不同的重复句（train）。
4. **清洗后同一模型的分数从 98.45% 掉到 97.90%，掉的全是它学错归属的部分——这正是"数据缺陷"的量化证据。**
5. 人工集引入了训练集中不存在的三类现象：越界问句（查他人考勤、CEO 薪酬）、部门边界问句
   （合规 vs 人事、合同文本 vs 客户场景）、词表外术语（"护目镜"在合成语料中出现 0 次）。
6. 结论：**对外汇报应给出三个数字——清洗后合成集 97.90%（模型对正确标签的水平）、
   人工集 91.53%（真实语义边界下的可用性）、以及未清洗 98.45%（数据缺陷导致的虚高），
   并说明差距来源。** 只报 98.45% 会在答辩时被问倒。

## 六、合成语料清洗（去重 + 重新分类）

### 6.1 缺陷证据

- **双标签主题 7 个**（同一主题字符串被两个类别各标约一半样本）：病假工资、育儿假、陪产假、
  加班费计算、保密协议、竞业协议、合同模板；另有「盖章流程」被 admin/legal 重复、
  「劳动仲裁」被 hr/legal 重复、「产假工资」「年假计算」被拆到 hr。
- **同文本异标签**：train.txt 1266 行、dev.txt 2 行、test.txt 4 行，是完全相同却带不同标签的重复句。
- 发现方法：按 Apriori 思路挖掘全语料频繁短语 -> 用"各类别均匀出现的片段"识别模板框架 ->
  剥掉框架得到主题词 -> 统计每个主题词的标签分布，最大占比 <= 0.90 者判为跨类主题。

### 6.2 分类依据（为什么这样裁定）

`data/knowledge_base/<部门>/*.md` 的 8 个部门目录与 `class.txt` 的 8 个业务类别一一对应
（policy/it/finance/hr/admin/engineering/legal/sales），第 9 类 `other_chitchat` 无知识库。
因此**主题归属 = 该主题在知识库中由哪个部门的文档承载**，每条规则的 evidence 都指向知识库原文：

| 关键裁定 | 依据 |
|----------|------|
| 病假工资 -> policy_attendance | `policy/attendance.md`："病假工资按基本工资的 80% 发放" |
| 育儿假 / 陪产假 -> policy_attendance | `policy/attendance.md` 2.4 其他假期："产假：158 天；陪产假：15 天"、"育儿假：子女 3 岁前每年各 5 天" |
| 加班费计算 -> policy_attendance | `policy/overtime.md` 章节标题《加班费计算》 |
| 保密协议 / 竞业协议 -> legal_contract | `legal/confidential.md` 章节《保密协议》《竞业限制》 |
| 合同模板 / 盖章 -> legal_contract | `legal/contract.md` 1.2 模板、《第二章 用章管理》 |
| 劳动仲裁 -> hr_onboarding | 知识库未收录，按裁定：离职劳动争议归人事，合同争议的仲裁条款归法务（`legal/contract.md` 第四章） |
| 产假工资 / 年假计算 -> policy_attendance | 与病假工资同属 `policy/attendance.md` 请假制度章节（概念归并，非矛盾标注） |

规则表在 `data/synthetic_label_rules.json`，每条含 `phrase / label / kind / evidence / before`，
`kind=conflict` 为必修的矛盾标注，`kind=concept_merge` 为可选的概念归并，便于组长复核与回滚。

### 6.3 清洗结果

| 文件 | 行数变化 | 重分类行数 | 去重丢弃 | 残留同文本异标签 |
|------|----------|------------|----------|------------------|
| `data/clean/train.txt` | 202500 -> 201234 | 3761 | 1266 | 0 |
| `data/clean/dev.txt` | 10000 -> 9998 | 190 | 2 | 0 |
| `data/clean/test.txt` | 11099 -> 11095 | 182 | 4 | 0 |

- 未命中规则的样本一律保持原标签：只清洗已确认有矛盾的地方，避免清洗本身引入新噪声。
- 原始文件保留在 `data/*.txt` 且已在 git 中，`data/clean/` 是新增目录，可随时对比与回滚。
- 仍存在**概念近邻但标签不矛盾**的主题（如 社保缴纳(hr) / 社保扣款(finance)、投影连接(it) / 投影借用(admin)），
  属部门职责的合理切分，未改动，留待组长决定是否进一步归并。
- 清洗报告（含每条规则命中数、类别分布前后对比）：`data/clean/train_report.md`、`dev_report.md`、`test_report.md`。

## 七、badcase 归类分析

完整分析见 `data/eval/badcase_notes.md`，并由 `src/evaluate.py` 自动并入每份 `badcases.txt` 的第三节，
内容包括：双标签主题的样本级证据、缩写捷径的机制、清洗前后的错误结构对比、人工集的错误明细、
以及 6 条按性价比排序的改进建议。

## 八、复现步骤

```powershell
# 1. 清洗合成语料（去重 + 按知识库重新分类）
python scripts/clean_synthetic.py --input data/train.txt --output data/clean/train.txt --report data/clean/train_report.md
python scripts/clean_synthetic.py --input data/dev.txt   --output data/clean/dev.txt   --report data/clean/dev_report.md
python scripts/clean_synthetic.py --input data/test.txt  --output data/clean/test.txt  --report data/clean/test_report.md

# 2. 三个数据集上的评估（产物会记录权重 sha256，换权重必须整套重跑）
python -m src.evaluate --input data/test.txt                --output-dir outputs/eval_test        # 未清洗基线
python -m src.evaluate --input data/clean/test.txt          --output-dir outputs/eval_test_clean  # 清洗后
python -m src.evaluate --input data/eval/test_queries.jsonl --output-dir outputs/eval_human      # 人工集

# 3. 生成对比表
python scripts/compare_eval.py --synthetic outputs/eval_test       --human outputs/eval_human --out docs/eval_comparison.md
python scripts/compare_eval.py --synthetic outputs/eval_test_clean --human outputs/eval_human --out docs/eval_comparison_clean.md
```

每次评估在输出目录产出四件套：`report.txt`（acc / macro-F1 / 分类报告）、`confusion_matrix.png`、
`badcases.txt`（混淆对总览 + 明细 + 人工归因 + 权重哈希）、`metrics.json`（机器可读，含 `ckpt_sha256`）。

## 九、验收对照

| 验收标准 | 状态 | 位置 |
|----------|------|------|
| report.txt（acc / macro-F1 / 分类报告） | 完成 | `outputs/eval_test/`、`outputs/eval_test_clean/`、`outputs/eval_human/` |
| confusion_matrix.png | 完成 | 同上三个目录，9x9 全类别（显式传 labels，避免类别缺席导致矩阵错位） |
| badcases.txt（混淆对归类 + 原因） | 完成 | 同上三个目录，含人工归因分析与权重哈希 |
| 人工评估集扩充到 100+ 条 | 完成 | `data/eval/test_queries.jsonl`，118 条，无重复，覆盖全部 9 类 |
| 合成集 vs 人工集差距结论 | 完成 | 本文第五节 + `docs/eval_comparison.md`、`docs/eval_comparison_clean.md` |
| 合成语料清洗（去重 + 重新分类） | 完成 | `scripts/clean_synthetic.py`、`data/synthetic_label_rules.json`、`data/clean/`（含三份清洗报告） |

## 十、过程中发现并修复的问题

1. **权重路径不一致**：`src/config.py` 原默认 `checkpoints/intent_bert/best.pt`，本机实际权重在 `models/best.pt`。
   已新增 `BEST_CKPT = models/best.pt` 并作为 `src/evaluate.py --ckpt` 的默认值。
   注意 `src/predict.py`、`prune.py`、`quantize.py`、`distill.py` 仍指向 `checkpoints/`，联调前需要统一。
2. **人工集标签越界**：`out_of_scope` 不在 `class.txt`，评估直接报"未知标签名"。已归一到 `other_chitchat`；
   同时把 evaluate 的报错信息改成"文件名 + 行号 + 合法标签位置"。
3. **缺本地预训练目录**：`models/` 下只有 `best.pt`，缺分词器与骨干配置，已按 README 约定补齐
   `models/bert-base-chinese/`（`models/` 已在 .gitignore 中，不入库）。
4. **badcases.txt 只有原始清单**：原实现按行输出 `真实\t预测\t文本`，无法回答"哪些类别容易混淆"。
   已重写为"混淆对总览 -> 明细（高频词证据 + 示例）-> 人工归因"三层结构。
5. **评估不产出机器可读结果**：新增 `metrics.json`，对比脚本因此不需要重跑模型即可复算结论。
6. **混淆矩阵维度不稳**：`confusion_matrix` 未指定 labels，某类在预测中缺席时矩阵会收缩、与标签名错位。
   已显式传 `labels=range(num_labels)`。
7. **合成语料矛盾标注**：7 个主题字符串被两个类别各标一半 + 1266 条同文本异标签重复行（详见第六节）。
   已提供可复核的清洗规则与清洗后的数据集。
8. **缩写捷径**：NDA/NPS/VPN 被字形特征带偏到 engineering_eq。属数据覆盖问题，需补缩写主题后重训，
   本次仅记录证据与建议，未改动标签（它们本身没有标错）。
9. **评估中途权重被替换（本次实测到）**：`models/best.pt` 于 2026-09-21 20:14 被覆盖，
   导致同一份测试集出现 0.9840 / 0.9845 两个分数、人工集出现 0.8898 / 0.9153 两个分数，
   而文件名完全一样、无法从产物判断指标出自哪份权重。
   已在 `src/evaluate.py` 中新增 `file_sha256()`，把权重哈希写入 `metrics.json` 与 `badcases.txt` 头部；
   **换权重后必须整套重跑**，报告与 PPT 里的数字要连同哈希一起记录。

## 十一、环境说明

本地复现使用的是 `E:\Anaconda\envs\TMF`（Python 3.10.21 + torch 2.14.0+cu126 + transformers 5.16.1，
RTX 3060 Laptop），与 `pyproject.toml` 的 `requires-python >=3.11,<3.12` 不一致：
该 conda 环境已带 CUDA 版 torch，而新建 3.11 环境需要重新下载数 GB 权重。
评估与清洗代码只使用 3.10 兼容语法，其余成员按 README 用 `uv venv --python 3.11 .venv` 建环境后可同样复现；
结论只依赖模型权重与数据，不依赖 Python 版本。
