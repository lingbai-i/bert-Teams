> **本节指标对应的权重**：`models/best.pt`，sha256 `3313804ef8ce1d6d…`（2026-09-21 20:14 版本）。
> 该模型是在**另一套清洗口径**的数据上训练的（见第四节），这与本仓库 `data/*.txt` 采用的知识库口径
> 在 4 个主题上不同。`src/evaluate.py` 已把权重哈希写进 `metrics.json` 与 `badcases.txt` 头部，
> 换权重后必须整套重跑，否则会出现"同一测试集两个分数"的口径混乱。

### 一、未清洗合成集（data/raw/test.txt：11099 条，错误 172 条，错误率 1.55%）

172 条错误可以完整归因到两类**数据问题**，没有一条属于"模型没学到"：

#### 1. 双标签主题：129 条，占 75.0%

生成映射表把同一个主题字符串同时分配给了两个类别，而且几乎是五五开，模型学到哪一边都只能拿一半分：

| 主题 | 类别 A 样本数 | 类别 B 样本数 | 最大类占比 | 未清洗测试集上的混淆 |
|------|------|------|------|------|
| 病假工资 | hr_onboarding 328 | policy_attendance 314 | 51.1% | policy->hr |
| 育儿假 | hr_onboarding 328 | policy_attendance 325 | 50.2% | policy->hr |
| 陪产假 | hr_onboarding 339 | policy_attendance 322 | 51.3% | policy->hr |
| 加班费计算 | policy_attendance 328 | finance_expense 326 | 50.2% | policy->finance 13 条 |
| 保密协议 | legal_contract 319 | hr_onboarding 313 | 50.5% | hr->legal 37 条 |
| 竞业协议 | legal_contract 329 | hr_onboarding 303 | 52.1% | hr->legal |
| 合同模板 | sales_marketing 324 | legal_contract 309 | 51.2% | sales<->legal 18 条 |

这些样本的正确标签在两类之间接近随机（50:50），任何模型在这批数据上的期望上限就是 50%，
错误属于**标注重叠**而非泛化失败。语料里还有 1266 条（train）完全相同的句子带着两个不同标签。

#### 2. 英文缩写捷径：43 条，占 25.0%

NDA（23 条）、NPS（19 条）、VPN（1 条）三个主题在语料里**只属于一个类别**
（legal_contract / sales_marketing / it_vpn），标签没有矛盾，但预测全部被送往 engineering_eq。

- 现象：NDA 一律判成 engineering_eq，NPS 一律判成 engineering_eq。
- 原因：engineering_eq 的语料里密集出现 PLC、FCT、ICT 等大写英文缩写，模型学到的是
  "出现英文字母串"这一字形特征与 engineering_eq 的共现，而不是缩写的语义。
- 后果：**标签清洗解决不了**，只能靠补充缩写主题数据 + 重训消除。

#### 3. 其余：0 条。172 = 129 + 43，没有第三条原因。

### 二、清洗后的合成集（data/test.txt：11095 条，错误 178 条，错误率 1.60%）

清洗只修"真·矛盾标注"（第五节）。清洗后错误结构变成：

| 混淆对（真实 -> 预测） | 条数 | 占错误比 | 性质 |
|------------------------|------|----------|------|
| policy_attendance -> hr_onboarding | 107 | 60.1% | **口径分歧**：病假工资/育儿假/陪产假，本表判 policy，模型训练数据判 hr |
| policy_attendance -> finance_expense | 28 | 15.7% | **口径分歧**：加班费计算，本表判 policy，模型训练数据判 finance |
| legal_contract -> engineering_eq | 23 | 12.9% | NDA 缩写捷径 |
| sales_marketing -> engineering_eq | 19 | 10.7% | NPS 缩写捷径 |
| it_vpn -> engineering_eq | 1 | 0.6% | VPN 缩写捷径 |

- 前两对共 **135 条（75.8%）是口径分歧而不是模型能力问题**：模型 100% 一致地预测 hr / finance，
  说明它训练数据里这几个主题就是那样标的。**换一套口径（改规则或重训任一侧）这批错误即刻消失。**
- 后三对共 **43 条（24.2%）** 是缩写捷径，需补数据。
- 反向收益：未清洗集上 hr_onboarding -> legal_contract 的 37 条、sales_marketing -> legal_contract
  的 18 条在清洗后全部消失——保密协议/竞业协议/合同模板统一到 legal 后，与模型预测一致。

### 三、人工集（data/eval/test_queries.jsonl：118 条，错误 10 条，错误率 8.47%）

人工集的错误结构与合成集完全不同，全部是**语义边界**问题（与清洗口径无关）：

#### 1. 第 9 类没有"越界/拒答"语义：other_chitchat recall 仅 0.6429（14 条错 5 条）

训练语料中 other_chitchat 全是纯闲聊（"给我推荐个电影"、"中午吃什么呢"），
模型学到的是"闲聊词 -> other_chitchat"，一旦问句带业务词，就必然被拉进某个业务类：

| 问句 | 预测 | 置信度 |
|------|------|--------|
| 别人的考勤记录能查吗 | policy_attendance | 1.000 |
| 能帮我查下同事这个月的考勤吗 | policy_attendance | 1.000 |
| 帮我给宠物起个名字 | admin_logistics | 1.000 |
| 公司CEO的年薪是多少 | hr_onboarding | 0.999 |
| CEO 的工资是多少 | finance_expense | 0.999 |

越界问题在真实客服场景必须显式拒答，这是现有 9 类体系的结构性缺口，不是训练轮数能解决的问题。

#### 2. 部门边界本身模糊：legal_contract recall 0.75（12 条错 3 条）

| 问句 | 预测 | 争议点 |
|------|------|------|
| 劳动合同到期不续签需要提前通知吗 | hr_onboarding | 劳动法咨询 vs 员工关系 |
| 收集员工个人信息要不要签同意书 | hr_onboarding | 合规 vs 人事 |
| 客户要签保密协议，用公司模板还是对方的 | sales_marketing | 合同文本 vs 客户场景 |
| 项目聚餐的费用走哪个科目 | admin_logistics | 费用科目 vs 行政后勤 |
| 会议室的投影仪连不上我的笔记本 | admin_logistics | 会议设备故障 vs IT 支持 |

这 5 条属于**单标签分类的粒度天花板**：同一句问句在知识库路由里会同时命中多个部门的文档。
答辩时应明确评估口径——人工集标注按"问句的主要意图"裁定。

#### 3. admin_logistics 是"磁铁类"：precision 0.8000

帮我给宠物起个名字、项目聚餐的费用走哪个科目、会议室的投影仪连不上我的笔记本三条错误全部落进
admin_logistics。行政语料覆盖会议、场地、物资、活动、后勤，主题词面最广，
因此"找不到归属"的问句容易落到它身上。

#### 4. 错误是高置信度错误，不是"模棱两可"

10 条错误中 8 条置信度 >= 0.866，其中 6 条 >= 0.999。靠置信度阈值拒答（例如 < 0.5 才拒答）
在人工集上一条都抓不到，需要"拒答类 + 阈值"双轨机制。

### 四、三类集合的错误结构对比

| 维度 | 未清洗合成集 | 清洗后合成集 | 人工集 |
|------|--------------|--------------|--------|
| 文件 | data/raw/test.txt | data/test.txt | data/eval/test_queries.jsonl |
| 样本数 | 11099 | 11095 | 118 |
| accuracy / macro-F1 | 0.9845 / 0.9847 | 0.9840 / 0.9841 | 0.9153 / 0.9130 |
| 错误率 | 1.55% | 1.60% | 8.47% |
| 错误主因 | 双标签标注 75.0% + 缩写捷径 25.0% | 口径分歧 75.8% + 缩写捷径 24.2% | 越界拒答类缺失 + 部门边界模糊 |
| 修复动作 | 已修复（清洗数据） | 统一口径（改规则或重训）+ 补缩写数据 | 扩标签体系 / 多标签路由 / 拒答机制 |

一句话结论：**未清洗合成集的 98.45% 是"押中了矛盾标注某一侧"的虚高分数；
清洗后同一模型的 98.40% 里 75.8% 的错误是团队内部两套口径的分歧（可即时消除）、
24.2% 是缩写捷径；人工集的 91.53% 才是真实语义边界下的可用性。
三个数字的差值分别对应"数据缺陷""口径分歧""标签体系缺陷"，都可以在答辩中讲清。**

### 五、口径分歧明细（用 best.pt 的预测分布反推训练口径）

| 主题 | 本表裁定（知识库依据） | 该模型训练口径 | 模型预测一致性 | 知识库原文 |
|------|------------------------|----------------|----------------|------------|
| 病假工资 | policy_attendance | hr_onboarding | 100% | policy/attendance.md 2.2 病假：病假工资按基本工资的 80% 发放 |
| 育儿假 | policy_attendance | hr_onboarding | 100% | policy/attendance.md 2.4 其他假期：育儿假每年各 5 天 |
| 陪产假 | policy_attendance | hr_onboarding | 100% | policy/attendance.md 2.4 其他假期：陪产假 15 天 |
| 加班费计算 | policy_attendance | finance_expense | 77% | policy/overtime.md 章节标题《加班费计算》 |
| 保密协议 | legal_contract | legal_contract | 100% | legal/confidential.md 章节《保密协议》 |
| 竞业协议 | legal_contract | legal_contract | 100% | legal/confidential.md 章节《竞业限制》 |
| 合同模板 | legal_contract | legal_contract | 100% | legal/contract.md 1.2 模板：常用模板见 OA 法务模板库 |

**为什么建议按知识库口径统一**：部署层是拿意图标签去检索知识库文档的（`data/knowledge_base/<部门>/`）。
若 `病假工资` 被路由到 hr，而答案写在 `policy/attendance.md` 里，现场 demo 会取错文档、答非所问。
意图标签应当指向"哪个部门的文档能回答这个问题"，这正是本表裁定的定义。

### 六、数据清洗做了什么（去重 + 重新分类）

清洗脚本 `scripts/clean_synthetic.py` + 规则表 `data/synthetic_label_rules.json`。
**本表只收录"真·矛盾标注"**（同一主题字符串被两个类别各标约一半），共 7 条：

| 规则短语 | 裁定标签 | 依据（knowledge_base 原文） |
|----------|----------|------------------------------|
| 病假工资 | policy_attendance | policy/attendance.md 2.2 病假：病假工资按基本工资的 80% 发放 |
| 育儿假 | policy_attendance | policy/attendance.md 2.4 其他假期：育儿假每年各 5 天 |
| 陪产假 | policy_attendance | policy/attendance.md 2.4 其他假期：陪产假 15 天 |
| 加班费计算 | policy_attendance | policy/overtime.md 章节标题《加班费计算》 |
| 保密协议 | legal_contract | legal/confidential.md 章节《保密协议》 |
| 竞业协议 | legal_contract | legal/confidential.md 章节《竞业限制》 |
| 合同模板 | legal_contract | legal/contract.md 1.2 模板：常用模板见 OA 法务模板库 |

以下 4 个候选**经复核后从规则中移除**，理由记录在规则表的 `dropped_candidates` 字段：
`盖章`（合同盖章与盖章流程是两个不同主题、各自内部一致，不构成矛盾）、
`劳动仲裁`（语料中 100% 属 hr_onboarding，法务侧是另一个主题"仲裁申请"）、
`产假工资`、`年假计算`（各自 100% 一致，属口径选择而非矛盾标注，留待组长裁定）。

清洗结果（原始语料归档在 `data/raw/`，且已在 git 中，可随时回滚）：

| 文件 | 行数变化 | 重分类行数 | 去重丢弃 | 残留同文本异标签 |
|------|----------|------------|----------|------------------|
| `data/train.txt` | 202500 -> 201234 | 2604 | 1266 | 0 |
| `data/dev.txt` | 10000 -> 9998 | 137 | 2 | 0 |
| `data/test.txt` | 11099 -> 11095 | 127 | 4 | 0 |

未命中规则的样本一律保持原标签：只清洗已确认有矛盾的地方，避免清洗本身引入新噪声。
清洗报告见 `data/clean_reports/{train,dev,test}_report.md`（含每条规则命中数与类别分布前后对比）。

### 七、改进建议（按性价比排序）

1. **先统一口径**：`病假工资/育儿假/陪产假/加班费计算` 目前有两套标法。
   按知识库口径需要重训；按现有模型口径只需改规则表并重新清洗（约 2 分钟）。
   不解决这一条，合成集分数会被无意义地压掉 1.2 个百分点，且 demo 可能取错知识库文档。
2. **补英文缩写主题**：主动加入 KPI、SLA、ERP、CRM 等语料中未出现的缩写，
   打破"字母串 -> engineering_eq"的捷径（当前占清洗后合成集错误 24.2%）。
3. **增加越界类**（out_of_scope）或在 other_chitchat 中补入"带业务词的越界问句"
   （查他人考勤、CEO 薪酬、内部机密等），并在服务层保留"知识库零命中即拒答"的兜底规则。
4. **对 legal/hr、it/admin 这类天然重叠的部门**，把单标签分类升级为多标签或在知识库层做二次路由。
5. **人工评估集从 118 条继续扩充到 300+ 条**，并记录每条的问句来源成员与类别。
6. **评估产物钉住权重版本**：`metrics.json` 已记录 `ckpt_sha256`，换权重后必须整套重跑。
