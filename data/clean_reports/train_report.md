# 合成语料清洗报告

输入: data\raw\train.txt
输出: data\train.txt

输入行数: 202500
重分类改动行数: 2604
去重丢弃行数: 1266
输出行数: 201234
残留同文本异标签数: 0（清洗目标为 0）

## 各规则命中情况

| 规则短语 | 裁定标签 | 类型 | 命中行数 | 依据 |
|----------|----------|------|----------|------|
| 加班费计算 | policy_attendance | conflict | 750 | knowledge_base/policy/overtime.md：章节标题《加班费计算》 |
| 病假工资 | policy_attendance | conflict | 733 | knowledge_base/policy/attendance.md：2.2 病假 —— 病假工资按基本工资的 80% 发放 |
| 保密协议 | legal_contract | conflict | 735 | knowledge_base/legal/confidential.md：章节标题《保密协议》 |
| 竞业协议 | legal_contract | conflict | 737 | knowledge_base/legal/confidential.md：章节标题《竞业限制》 |
| 合同模板 | legal_contract | conflict | 743 | knowledge_base/legal/contract.md：1.2 模板 —— 常用模板见 OA“法务模板库” |
| 育儿假 | policy_attendance | conflict | 751 | knowledge_base/policy/attendance.md：2.4 其他假期 —— 育儿假：子女 3 岁前夫妻双方每年各 5 天 |
| 陪产假 | policy_attendance | conflict | 746 | knowledge_base/policy/attendance.md：2.4 其他假期 —— 产假：158 天；陪产假：15 天 |

## 类别分布（清洗前 -> 清洗后）

| 类别 | 清洗前 | 清洗后 | 变化 |
|------|--------|--------|------|
| policy_attendance | 22500 | 23288 | +788 |
| it_vpn | 22500 | 22500 | +0 |
| finance_expense | 22500 | 22124 | -376 |
| hr_onboarding | 22500 | 20652 | -1848 |
| admin_logistics | 22500 | 22500 | +0 |
| engineering_eq | 22500 | 22500 | +0 |
| legal_contract | 22500 | 23050 | +550 |
| sales_marketing | 22500 | 22120 | -380 |
| other_chitchat | 22500 | 22500 | +0 |
