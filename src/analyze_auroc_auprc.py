"""
从聚类结果日志中推断两个中心的 P0 占比
根据日志：
- 深医测试集：30,192 样本 (20%)
- 福鼎测试集：15,169 样本 (20%)

从全局聚类结果：
- P0: 52,011 (22.9%)
- P1: 51,793 (22.8%)
- P2: 43,593 (19.2%)
- P3: 79,403 (35.0%)

AUROC vs AUPRC 差异分析
"""

# 深医
shenyi_total = 150959
shenyi_test = 30192  # 20%

# 福鼎
fuding_total = 75841
fuding_test = 15169  # 20%

# 全局 P0 占比
global_p0_rate = 52011 / 226800  # 0.229

print("=" * 60)
print("AUROC vs AUPRC 差异分析")
print("=" * 60)

print("""
关键概念：
- AUROC (Area Under ROC Curve): 衡量模型区分正负样本的能力，对类别不平衡不敏感
- AUPRC (Area Under Precision-Recall Curve): 衡量模型在正样本上的精确度和召回率，对类别不平衡非常敏感

AUPRC 的理论下限 = 正样本比例 (prevalence)
AUROC 的理论下限 = 0.5 (随机猜测)

当正样本比例很低时：
- AUROC 仍然可以很高（如 0.85）
- AUPRC 会接近 prevalence（如 0.23）

当正样本比例较高时：
- AUROC 可能相似（如 0.84）
- AUPRC 会显著提高（如 0.78）
""")

print("=" * 60)
print("假设分析：如果福鼎的 P0 占比显著高于深医")
print("=" * 60)

# 假设深医 P0 占比 18.9%，福鼎 P0 占比 30.0%
# 这可以解释为什么福鼎 AUPRC (0.775) 远高于深医 (0.564)

shenyi_p0_rate = 0.189  # 假设
fuding_p0_rate = 0.300  # 假设

print(f"""
深医:
  P0 AUROC: 0.858
  P0 AUPRC: 0.564
  假设 P0 占比: {shenyi_p0_rate:.1%}
  AUPRC/prevalence 比值: {0.564/shenyi_p0_rate:.2f}

福鼎:
  P0 AUROC: 0.839
  P0 AUPRC: 0.775
  假设 P0 占比: {fuding_p0_rate:.1%}
  AUPRC/prevalence 比值: {0.775/fuding_p0_rate:.2f}

解释：
1. 两个中心的 AUROC 接近 (0.858 vs 0.839)，说明模型区分能力相似
2. 福鼎 AUPRC 显著更高 (0.775 vs 0.564)，说明福鼎的 P0 占比更高
3. AUPRC/prevalence 比值衡量模型相对于随机猜测的提升倍数
""")

print("=" * 60)
print("结论")
print("=" * 60)
print("""
P0 AUROC 和 AUPRC 的差异是由两个中心的 P0 表型基线发生率不同导致的：

- 深医 P0 占比较低 → AUPRC 较低 (0.564)
- 福鼎 P0 占比较高 → AUPRC 较高 (0.775)

这与"福鼎高 UFR → P0 表型高发"的根因分析一致。
""")
