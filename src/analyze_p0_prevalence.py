"""
分析两个中心 P0 表型占比，解释 AUROC vs AUPRC 差异
"""
import pickle
import numpy as np
from pathlib import Path

PROCESSED_DIR = Path("/home/cht/Works/OptiHemoAI/hd_trajectory_poc/data/processed")

# 查找包含 phenotype_id 的文件
for fp in PROCESSED_DIR.glob("*.pkl"):
    with open(fp, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, list) and len(data) > 0:
        if "phenotype_id" in data[0]:
            print(f"\n找到包含 phenotype_id 的文件: {fp.name}")
            all_labeled = data
            break
else:
    print("未找到包含 phenotype_id 的文件")
    exit(1)

for center, center_name in [("shenyi", "深医"), ("fuding", "福鼎")]:
    cs = [s for s in all_labeled if s["center"] == center]
    n_total = len(cs)
    
    p_counts = {}
    for k in range(4):
        p_counts[k] = sum(1 for s in cs if s.get("phenotype_id") == k)
    
    print(f"\n{center_name} (总样本: {n_total}):")
    for k in range(4):
        pct = p_counts[k] / n_total * 100
        print(f"  P{k}: {p_counts[k]} ({pct:.1f}%)")
    
    p0_rate = p_counts[0] / n_total
    print(f"  P0 基线发生率 (prevalence): {p0_rate:.4f}")
    
    # AUPRC 的理论下限是 prevalence
    print(f"  AUPRC 随机基线 (prevalence): {p0_rate:.4f}")
    print(f"  AUROC 随机基线: 0.5000")
