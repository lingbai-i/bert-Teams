# 合成集 vs 人工集 评估对比

> 由 `python scripts/compare_eval.py` 依据两侧 metrics.json 自动生成，请勿手工修改数字。

## 一、总体指标

| 指标 | 合成集 | 人工集 | 差值（人工 - 合成） |
|------|--------|--------|---------------------|
| 样本数 | 11095 | 118 | -10977 |
| accuracy | 0.9790 | 0.9153 | -6.37pp |
| macro-F1 | 0.9792 | 0.9130 | -6.62pp |
| weighted-F1 | 0.9789 | 0.9121 | -6.68pp |
| bad case 数 | 233 | 10 | -223 |
| 错误率 | 0.0210 | 0.0847 | +6.37pp |

## 二、逐类 F1（按人工集 F1 升序，人工集上表现最弱的类别排在最前）

| 类别 | 合成集 F1 | 人工集 F1 | 差值 | 判定 |
|------|-----------|-----------|------|------|
| other_chitchat | 1.0000 | 0.7826 | -21.74pp | 退化 |
| legal_contract | 0.9820 | 0.8571 | -12.49pp | 退化 |
| admin_logistics | 0.9903 | 0.8889 | -10.14pp | 退化 |
| hr_onboarding | 0.9425 | 0.9032 | -3.93pp | 退化 |
| finance_expense | 0.9887 | 0.9286 | -6.01pp | 退化 |
| policy_attendance | 0.9345 | 0.9333 | -0.12pp | 持平 |
| sales_marketing | 0.9922 | 0.9600 | -3.22pp | 退化 |
| it_vpn | 0.9996 | 0.9630 | -3.66pp | 退化 |
| engineering_eq | 0.9831 | 1.0000 | +1.69pp | 提升 |

## 三、主要混淆对（各取前 5 对）

### 合成集（错误数 233，错误率 2.10%）

| 混淆对（真实 -> 预测） | 条数 | 占错误比 |
|------------------------|------|----------|
| policy_attendance -> hr_onboarding | 138 | 59.2% |
| policy_attendance -> finance_expense | 28 | 12.0% |
| legal_contract -> admin_logistics | 24 | 10.3% |
| legal_contract -> engineering_eq | 23 | 9.9% |
| sales_marketing -> engineering_eq | 19 | 8.2% |

### 人工集（错误数 10，错误率 8.47%）

| 混淆对（真实 -> 预测） | 条数 | 占错误比 |
|------------------------|------|----------|
| legal_contract -> hr_onboarding | 2 | 20.0% |
| other_chitchat -> policy_attendance | 2 | 20.0% |
| finance_expense -> admin_logistics | 1 | 10.0% |
| it_vpn -> admin_logistics | 1 | 10.0% |
| legal_contract -> sales_marketing | 1 | 10.0% |

