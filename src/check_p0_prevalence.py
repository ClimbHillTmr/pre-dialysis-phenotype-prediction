"""
计算两个中心各自的 P0 表型占比，解释 AUROC vs AUPRC 差异
"""
import pickle
import numpy as np
from pathlib import Path

PROCESSED_DIR = Path("/home/cht/Works/OptiHemoAI/hd_trajectory_poc/data/processed")

with open(PROCESSED_DIR / "all_cleaned_full.pkl", "rb") as f:
    all_labeled = pickle.load(f)

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
    print(f"  P0 基线发生率: {p0_rate:.4f}")
