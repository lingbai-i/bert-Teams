# 合成语料清洗报告

输入: data\raw\test.txt
输出: data\test.txt

输入行数: 11099
重分类改动行数: 130
去重丢弃行数: 4
输出行数: 11095
残留同文本异标签数: 0（清洗目标为 0）

## 各规则命中情况

| 规则短语 | 裁定标签 | 类型 | 命中行数 | 裁定依据 | knowledge_base 对照 |
|----------|----------|------|----------|----------|---------------------|
| 加班费计算 | finance_expense | conflict | 39 | convention_decision | knowledge_base/policy/overtime.md 的章节标题即《加班费计算》；本表按团队口径决策归入 finance_expense |
| 病假工资 | hr_onboarding | conflict | 46 | convention_decision | knowledge_base/policy/attendance.md 2.2 病假同时写了病假工资的发放比例；本表按团队口径决策归入薪酬福利侧（hr_onboarding） |
| 保密协议 | legal_contract | conflict | 33 | knowledge_base | knowledge_base/legal/confidential.md 章节标题《保密协议》，两套口径一致 |
| 竞业协议 | legal_contract | conflict | 43 | knowledge_base | knowledge_base/legal/confidential.md 章节标题《竞业限制》，两套口径一致 |
| 合同模板 | legal_contract | conflict | 39 | knowledge_base | knowledge_base/legal/contract.md 1.2 模板：常用模板见 OA 法务模板库，两套口径一致 |
| 育儿假 | hr_onboarding | conflict | 37 | convention_decision | knowledge_base/policy/attendance.md 2.4 其他假期列有育儿假；本表按团队口径决策归入 hr_onboarding |
| 陪产假 | hr_onboarding | conflict | 25 | convention_decision | knowledge_base/policy/attendance.md 2.4 其他假期列有陪产假；本表按团队口径决策归入 hr_onboarding |

## 类别分布（清洗前 -> 清洗后）

| 类别 | 清洗前 | 清洗后 | 变化 |
|------|--------|--------|------|
| policy_attendance | 1250 | 1175 | -75 |
| it_vpn | 1250 | 1250 | +0 |
| finance_expense | 1250 | 1266 | +16 |
| hr_onboarding | 1250 | 1269 | +19 |
| admin_logistics | 1250 | 1250 | +0 |
| engineering_eq | 1250 | 1250 | +0 |
| legal_contract | 1250 | 1304 | +54 |
| sales_marketing | 1250 | 1232 | -18 |
| other_chitchat | 1099 | 1099 | +0 |
