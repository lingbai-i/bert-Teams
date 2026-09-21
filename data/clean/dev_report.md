# 合成语料清洗报告

输入: data\dev.txt
输出: data\clean\dev.txt

输入行数: 10000
重分类改动行数: 190
去重丢弃行数: 2
输出行数: 9998
残留同文本异标签数: 0（清洗目标为 0）

## 各规则命中情况

| 规则短语 | 裁定标签 | 类型 | 命中行数 | 依据 |
|----------|----------|------|----------|------|
| 加班费计算 | policy_attendance | conflict | 46 | knowledge_base/policy/overtime.md：章节标题《加班费计算》 |
| 病假工资 | policy_attendance | conflict | 26 | knowledge_base/policy/attendance.md：2.2 病假 —— 病假工资按基本工资的 80% 发放 |
| 保密协议 | legal_contract | conflict | 44 | knowledge_base/legal/confidential.md：章节标题《保密协议》 |
| 竞业协议 | legal_contract | conflict | 43 | knowledge_base/legal/confidential.md：章节标题《竞业限制》 |
| 合同模板 | legal_contract | conflict | 46 | knowledge_base/legal/contract.md：1.2 模板 —— 常用模板见 OA“法务模板库” |
| 劳动仲裁 | hr_onboarding | conflict | 19 | 知识库未收录该主题。按裁定口径：离职劳动争议归人事（hr/onboarding.md 第二章转正与离职），合同争议的仲裁条款归法务（legal/contract.md 第四章争议处理），故“劳动仲裁”判给 hr_onboarding，“仲裁申请”仍留 legal_contract |
| 产假工资 | policy_attendance | concept_merge | 15 | knowledge_base/policy/attendance.md：2.4 其他假期（产假/陪产假/婚假/丧假）与 2.2 病假工资同属请假制度章节，假期待遇统一归 policy_attendance |
| 年假计算 | policy_attendance | concept_merge | 25 | knowledge_base/policy/attendance.md：2.1 年假 —— 入职满 1 年享 5 天年假；其余年假主题（年假天数/年假折算/离职年假/年假有效期）均已是 policy_attendance，此处统一 |
| 育儿假 | policy_attendance | conflict | 47 | knowledge_base/policy/attendance.md：2.4 其他假期 —— 育儿假：子女 3 岁前夫妻双方每年各 5 天 |
| 陪产假 | policy_attendance | conflict | 43 | knowledge_base/policy/attendance.md：2.4 其他假期 —— 陪产假：15 天 |
| 盖章 | legal_contract | conflict | 33 | knowledge_base/legal/contract.md：第二章 用章管理 —— 《用章申请》→ 部门审批 → 法务审核 → 到行政部盖章 |

## 类别分布（清洗前 -> 清洗后）

| 类别 | 清洗前 | 清洗后 | 变化 |
|------|--------|--------|------|
| policy_attendance | 1250 | 1372 | +122 |
| it_vpn | 1250 | 1250 | +0 |
| finance_expense | 1250 | 1230 | -20 |
| hr_onboarding | 1250 | 1111 | -139 |
| admin_logistics | 1250 | 1237 | -13 |
| engineering_eq | 1250 | 1250 | +0 |
| legal_contract | 1250 | 1316 | +66 |
| sales_marketing | 1250 | 1232 | -18 |
