# 合成集 vs 人工集 评估对比

> 由 `python scripts/compare_eval.py` 依据两侧 metrics.json 自动生成，请勿手工修改数字。

## 一、总体指标

| 指标 | 合成集 | 人工集 | 差值（人工 - 合成） |
|------|--------|--------|---------------------|
| 样本数 | 11099 | 118 | -10981 |
| accuracy | 0.9840 | 0.8898 | -9.42pp |
| macro-F1 | 0.9841 | 0.8856 | -9.85pp |
| weighted-F1 | 0.9839 | 0.8865 | -9.74pp |
| bad case 数 | 178 | 13 | -165 |
| 错误率 | 0.0160 | 0.1102 | +9.42pp |

## 二、逐类 F1（按人工集 F1 升序，人工集上表现最弱的类别排在最前）

| 类别 | 合成集 F1 | 人工集 F1 | 差值 | 判定 |
|------|-----------|-----------|------|------|
| other_chitchat | 1.0000 | 0.7826 | -21.74pp | 退化 |
| legal_contract | 0.9677 | 0.8000 | -16.77pp | 退化 |
| admin_logistics | 1.0000 | 0.8148 | -18.52pp | 退化 |
| hr_onboarding | 0.9627 | 0.9032 | -5.95pp | 退化 |
| sales_marketing | 0.9839 | 0.9231 | -6.08pp | 退化 |
| it_vpn | 0.9996 | 0.9286 | -7.10pp | 退化 |
| finance_expense | 0.9921 | 0.9286 | -6.35pp | 退化 |
| policy_attendance | 0.9683 | 0.9333 | -3.50pp | 退化 |
| engineering_eq | 0.9831 | 0.9565 | -2.66pp | 退化 |

## 三、主要混淆对（各取前 5 对）

### 合成集（错误数 178，错误率 1.60%）

| 混淆对（真实 -> 预测） | 条数 | 占错误比 |
|------------------------|------|----------|
| policy_attendance -> hr_onboarding | 57 | 32.0% |
| hr_onboarding -> legal_contract | 37 | 20.8% |
| legal_contract -> engineering_eq | 23 | 12.9% |
| sales_marketing -> engineering_eq | 19 | 10.7% |
| policy_attendance -> finance_expense | 18 | 10.1% |

### 人工集（错误数 13，错误率 11.02%）

| 混淆对（真实 -> 预测） | 条数 | 占错误比 |
|------------------------|------|----------|
| legal_contract -> hr_onboarding | 2 | 15.4% |
| other_chitchat -> policy_attendance | 2 | 15.4% |
| admin_logistics -> sales_marketing | 1 | 7.7% |
| engineering_eq -> admin_logistics | 1 | 7.7% |
| finance_expense -> admin_logistics | 1 | 7.7% |

