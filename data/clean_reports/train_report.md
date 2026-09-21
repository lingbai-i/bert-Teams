# 合成语料清洗报告

输入: data\raw\train.txt
输出: data\train.txt

输入行数: 202500
重分类改动行数: 2584
去重丢弃行数: 1266
输出行数: 201234
残留同文本异标签数: 0（清洗目标为 0）

## 各规则命中情况

| 规则短语 | 裁定标签 | 类型 | 命中行数 | 裁定依据 | knowledge_base 对照 |
|----------|----------|------|----------|----------|---------------------|
| 加班费计算 | finance_expense | conflict | 750 | convention_decision | knowledge_base/policy/overtime.md 的章节标题即《加班费计算》；本表按团队口径决策归入 finance_expense |
| 病假工资 | hr_onboarding | conflict | 733 | convention_decision | knowledge_base/policy/attendance.md 2.2 病假同时写了病假工资的发放比例；本表按团队口径决策归入薪酬福利侧（hr_onboarding） |
| 保密协议 | legal_contract | conflict | 735 | knowledge_base | knowledge_base/legal/confidential.md 章节标题《保密协议》，两套口径一致 |
| 竞业协议 | legal_contract | conflict | 737 | knowledge_base | knowledge_base/legal/confidential.md 章节标题《竞业限制》，两套口径一致 |
| 合同模板 | legal_contract | conflict | 743 | knowledge_base | knowledge_base/legal/contract.md 1.2 模板：常用模板见 OA 法务模板库，两套口径一致 |
| 育儿假 | hr_onboarding | conflict | 751 | convention_decision | knowledge_base/policy/attendance.md 2.4 其他假期列有育儿假；本表按团队口径决策归入 hr_onboarding |
| 陪产假 | hr_onboarding | conflict | 746 | convention_decision | knowledge_base/policy/attendance.md 2.4 其他假期列有陪产假；本表按团队口径决策归入 hr_onboarding |

## 类别分布（清洗前 -> 清洗后）

| 类别 | 清洗前 | 清洗后 | 变化 |
|------|--------|--------|------|
| policy_attendance | 22500 | 21020 | -1480 |
| it_vpn | 22500 | 22500 | +0 |
| finance_expense | 22500 | 22700 | +200 |
| hr_onboarding | 22500 | 22344 | -156 |
| admin_logistics | 22500 | 22500 | +0 |
| engineering_eq | 22500 | 22500 | +0 |
| legal_contract | 22500 | 23050 | +550 |
| sales_marketing | 22500 | 22120 | -380 |
| other_chitchat | 22500 | 22500 | +0 |
