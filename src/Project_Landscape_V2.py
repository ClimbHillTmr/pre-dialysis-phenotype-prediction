"""
Project Landscape V2: 30万全量数据 + 顶刊发表图表 (Optimized with Parallel DTW)
====================================================
Phase 0: 全量数据清洗 (新增超滤量/IDWG字段)
Phase 1: 两阶段聚类 (50K采样→Centroids→25万分配)
Phase 2: Root Cause Analysis (UFR/IDWG差异检验)
Phase 3: 全量LightGBM + SHAP解释性

Author: Clinical Data Analytics Specialist
Date: 2026-04-16
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.stats import chi2_contingency, mannwhitneyu, ttest_ind
from scipy.ndimage import gaussian_filter1d
from fastdtw import fastdtw
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
import lightgbm as lgb
import shap
import warnings
from joblib import Parallel, delayed
import multiprocessing

warnings.filterwarnings("ignore")
plt.rcParams["font.size"] = 10
plt.rcParams["figure.dpi"] = 150

PROJECT_ROOT = Path("/home/cht/Works/OptiHemoAI/hd_trajectory_poc")
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

MAX_SEQ_LEN = 51
N_CLUSTERS = 4
RANDOM_STATE = 42
SAMPLE_SIZE = 50_000
TOTAL_ESTIMATED = 250_000
N_JOBS = max(1, multiprocessing.cpu_count() - 2)


# =============================================================================
# UFR语义溯源函数
# =============================================================================


def compute_ufr_fields(row, center):
    """
    计算UFR相关字段 (深医和福鼎语义不同)

    深医:
      - 透析前预设UFV: 处方超滤总量 (L)
      - 超滤率: 时间序列 (ml/hr)
      - 超滤量: 时间序列 (ml, 累计)

    福鼎:
      - UFR: 处方超滤率 (L/hr) - 已验证: UFR*时长 ≈ 体重丢失
      - 无时间序列超滤率
    """
    result = {"prescribed_ufr": np.nan, "actual_ufr": np.nan}

    try:
        duration = float(row.get("实际透析时长", np.nan))
        if duration <= 0 or np.isnan(duration):
            return result
    except:
        return result

    if center == "shenyi":
        # 处方UFR: 透析前预设UFV (L) / 实际时长 (hr) → ml/hr
        try:
            prescribed_ufv = float(row.get("透析前预设UFV", np.nan))
            if prescribed_ufv > 0:
                result["prescribed_ufr"] = (prescribed_ufv * 1000) / duration  # L→ml
        except:
            pass

        # 实际UFR: 超滤量 (ml) / 实际时长 (hr)
        try:
            uf_volume_str = str(row.get("超滤量", ""))
            if uf_volume_str and uf_volume_str != "nan":
                import ast

                uf_list = ast.literal_eval(uf_volume_str)
                if isinstance(uf_list, list):
                    # 取最后一个非NA值作为总超滤量
                    for val in reversed(uf_list):
                        if val != "NA" and val is not None:
                            uf_volume = float(val)
                            if uf_volume > 0:
                                result["actual_ufr"] = uf_volume / duration
                            break
        except:
            pass

    elif center == "fuding":
        # 福鼎UFR字段是 L/hr，转换为 ml/hr
        try:
            ufr_l_hr = float(row.get("UFR", np.nan))
            if ufr_l_hr > 0:
                result["prescribed_ufr"] = ufr_l_hr * 1000  # L/hr → ml/hr
                result["actual_ufr"] = ufr_l_hr * 1000  # 福鼎无实际值，用处方值近似
        except:
            pass

    return result


def compute_idwg(row):
    """计算透间体重增长: 透前体重 - 干体重 (kg)"""
    try:
        pre_weight = float(row.get("透前体重", np.nan))
        dry_weight = float(row.get("干体重", np.nan))
        if pre_weight > 0 and dry_weight > 0:
            return pre_weight - dry_weight
    except:
        pass
    return np.nan


# =============================================================================
# 数据清洗 (复用V2逻辑 + 新增UFR相关字段)
# =============================================================================


def convert_to_float_list(input_list):
    if isinstance(input_list, str):
        try:
            parsed_list = eval(input_list)
            if isinstance(parsed_list, list):
                return [
                    float(v)
                    for v in parsed_list
                    if isinstance(v, (int, float, str)) and v and v != "NA"
                ]
        except:
            pass
    elif isinstance(input_list, list):
        return [
            float(v)
            for v in input_list
            if isinstance(v, (int, float, str)) and v and v != "NA"
        ]
    return []


def validate_bp(sbp, dbp=None):
    if sbp is None or np.isnan(sbp) or sbp < 40 or sbp > 300:
        return False
    if dbp is not None and not np.isnan(dbp):
        if dbp < 20 or dbp > 150 or sbp <= dbp:
            return False
    return True


def validate_hr(hr):
    if hr is None or np.isnan(hr):
        return True
    return 30 <= hr <= 250


def validate_ufr(ufr):
    if ufr is None or np.isnan(ufr):
        return True
    return 0 <= ufr <= 2000


def parse_shenyi_full(row):
    """解析深医完整记录(含UFR相关字段)"""
    try:
        ufr_fields = compute_ufr_fields(row, "shenyi")
        session = {
            "center": "shenyi",
            "patient_id": str(row["患者id"]),
            "session_id": str(row["透析记录id"]),
            "dialysis_date": row["透析日期"],
            "age": row.get("年龄", np.nan),
            "sex": row.get("性别", "未知"),
            "prescribed_ufr": ufr_fields["prescribed_ufr"],
            "actual_ufr": ufr_fields["actual_ufr"],
            "idwg": compute_idwg(row),
        }

        pre_bp = str(row.get("透前血压", ""))
        if "/" in pre_bp:
            parts = pre_bp.split("/")
            session["pre_sbp"] = float(parts[0]) if parts[0] else np.nan
            session["pre_dbp"] = (
                float(parts[1]) if len(parts) > 1 and parts[1] else np.nan
            )
        else:
            session["pre_sbp"] = np.nan
            session["pre_dbp"] = np.nan

        mid_sbp_list = convert_to_float_list(row.get("透析中收缩压", []))
        mid_dbp_list = convert_to_float_list(row.get("透析中舒张压", []))
        mid_hr_list = convert_to_float_list(row.get("透析中脉搏", []))
        ufr_list = convert_to_float_list(row.get("超滤率", []))

        post_bp = str(row.get("透析后血压", ""))
        if "/" in post_bp:
            parts = post_bp.split("/")
            session["post_sbp"] = float(parts[0]) if parts[0] else np.nan
            session["post_dbp"] = (
                float(parts[1]) if len(parts) > 1 and parts[1] else np.nan
            )
        else:
            session["post_sbp"] = np.nan
            session["post_dbp"] = np.nan

        (
            session["time_sbp"],
            session["time_dbp"],
            session["time_hr"],
            session["time_ufr"],
        ) = ([], [], [], [])

        if validate_bp(session["pre_sbp"], session["pre_dbp"]):
            session["time_sbp"].append(session["pre_sbp"])
            session["time_dbp"].append(session["pre_dbp"])
            session["time_hr"].append(np.nan)
            session["time_ufr"].append(ufr_list[0] if ufr_list else np.nan)

        for i, sbp in enumerate(mid_sbp_list):
            dbp = mid_dbp_list[i] if i < len(mid_dbp_list) else np.nan
            hr = mid_hr_list[i] if i < len(mid_hr_list) else np.nan
            ufr = ufr_list[i] if i < len(ufr_list) else np.nan
            if validate_bp(sbp, dbp):
                session["time_sbp"].append(sbp)
                session["time_dbp"].append(dbp)
                session["time_hr"].append(hr if validate_hr(hr) else np.nan)
                session["time_ufr"].append(ufr if validate_ufr(ufr) else np.nan)

        if validate_bp(session["post_sbp"], session["post_dbp"]):
            session["time_sbp"].append(session["post_sbp"])
            session["time_dbp"].append(session["post_dbp"])
            session["time_hr"].append(row.get("透后脉搏", np.nan))
            session["time_ufr"].append(np.nan)

        session["dialysis_duration"] = row.get("实际透析时长", 4.0)
        session["dry_weight"] = row.get("干体重", np.nan)
        session["pre_weight"] = row.get("透前体重", np.nan)
        session["n_timepoints"] = len(session["time_sbp"])

        if session["n_timepoints"] < 5:
            return None
        return session
    except:
        return None


def parse_fuding_full(row, pre_post_df):
    """解析福鼎完整记录(含UFR相关字段)"""
    import re

    try:
        patient_name = str(row.get("NAME", row.get("姓名", "")))
        dialysis_date_raw = str(row.get("透析日期", ""))

        date_match = re.search(r"(\d+)年(\d+)月(\d+)日", dialysis_date_raw)
        if not date_match:
            return None
        year_month = f"{date_match.group(1)}年{date_match.group(2)}月"

        name_col = "NAME" if "NAME" in pre_post_df.columns else "姓名"
        pp_match = pre_post_df[
            (pre_post_df[name_col] == patient_name)
            & (pre_post_df["透析日期"].astype(str).str.contains(year_month, na=False))
        ]
        if len(pp_match) == 0:
            return None

        pp_row = pp_match.iloc[0]
        dialysis_date = dialysis_date_raw

        ufr_fields = compute_ufr_fields(row, "fuding")
        session = {
            "center": "fuding",
            "patient_id": patient_name,
            "session_id": f"{patient_name}_{dialysis_date}",
            "dialysis_date": dialysis_date,
            "age": row.get("AGE", np.nan),
            "sex": "未知",
            "prescribed_ufr": ufr_fields["prescribed_ufr"],
            "actual_ufr": ufr_fields["actual_ufr"],
            "idwg": compute_idwg(row),
        }

        pre_sbp_val = pp_row.get("透前收缩压", np.nan)
        pre_dbp_val = pp_row.get("透前舒张压", np.nan)
        session["pre_sbp"] = float(pre_sbp_val) if not pd.isna(pre_sbp_val) else np.nan
        session["pre_dbp"] = float(pre_dbp_val) if not pd.isna(pre_dbp_val) else np.nan

        post_sbp_val = pp_row.get("透后收缩压", np.nan)
        post_dbp_val = pp_row.get("透后舒张压", np.nan)
        session["post_sbp"] = (
            float(post_sbp_val) if not pd.isna(post_sbp_val) else np.nan
        )
        session["post_dbp"] = (
            float(post_dbp_val) if not pd.isna(post_dbp_val) else np.nan
        )

        (
            session["time_sbp"],
            session["time_dbp"],
            session["time_hr"],
            session["time_ufr"],
        ) = ([], [], [], [])

        if validate_bp(session["pre_sbp"], session["pre_dbp"]):
            session["time_sbp"].append(session["pre_sbp"])
            session["time_dbp"].append(session["pre_dbp"])
            session["time_hr"].append(np.nan)
            session["time_ufr"].append(np.nan)

        pressure_str = str(row.get("PRESSURE", ""))
        if pressure_str and pressure_str != "nan":
            readings = pressure_str.split(",")
            for reading in readings:
                if "\\" in reading:
                    parts = reading.split("\\")
                    try:
                        sbp = float(parts[0])
                        dbp = float(parts[1]) if len(parts) > 1 and parts[1] else np.nan
                        if validate_bp(sbp, dbp):
                            session["time_sbp"].append(sbp)
                            session["time_dbp"].append(dbp)
                    except:
                        continue

        hr_str = str(row.get("CARDIOTACH", ""))
        session["time_hr"] = [np.nan] * len(session["time_sbp"])
        if hr_str and hr_str != "nan":
            hr_values = hr_str.split(",")
            temp_hr = []
            for h in hr_values:
                try:
                    val = float(h) if h and h != "0" else np.nan
                    temp_hr.append(val if validate_hr(val) else np.nan)
                except:
                    temp_hr.append(np.nan)
            while len(temp_hr) < len(session["time_sbp"]):
                temp_hr.append(np.nan)
            session["time_hr"] = temp_hr[: len(session["time_sbp"])]

        if validate_bp(session["post_sbp"], session["post_dbp"]):
            session["time_sbp"].append(session["post_sbp"])
            session["time_dbp"].append(session["post_dbp"])
            session["time_hr"].append(np.nan)
            session["time_ufr"].append(np.nan)

        session["dialysis_duration"] = float(row.get("实际透析时长", 4.0))
        session["dry_weight"] = pp_row.get("干体重", np.nan)
        session["pre_weight"] = pp_row.get("透前体重", np.nan)
        session["n_timepoints"] = len(session["time_sbp"])

        if session["n_timepoints"] < 5:
            return None
        return session
    except:
        return None


def validate_session_completeness(session, min_tp=5):
    if session["n_timepoints"] < min_tp:
        return False
    sbp_valid = sum(1 for x in session["time_sbp"] if not np.isnan(x))
    if sbp_valid < min_tp:
        return False
    return True


def extract_idh_label(session, threshold_abs=90.0):
    """
    使用临床 IDH 定义: 透析中 SBP < 90 mmHg (绝对阈值)
    避免与 ΔSBP 聚类定义的循环论证
    """
    min_sbp = min(
        [x for x in session["time_sbp"][1:] if not np.isnan(x)], default=np.nan
    )
    if np.isnan(min_sbp):
        return "unknown"
    return "IDH" if min_sbp < threshold_abs else "Stable"


def normalize_to_deltasbp(session):
    time_sbp = np.array(session["time_sbp"])
    if len(time_sbp) < 5:
        return None
    baseline_sbp = time_sbp[0]
    if baseline_sbp <= 0 or np.isnan(baseline_sbp):
        return None
    delta_sbp = time_sbp - baseline_sbp
    time_rel = np.linspace(0, 1, len(delta_sbp))
    time_tgt = np.linspace(0, 1, MAX_SEQ_LEN)
    try:
        interp = interp1d(time_rel, delta_sbp, kind="linear", fill_value="extrapolate")
        return interp(time_tgt)
    except:
        return None


def smooth_traj(traj, sigma=1):
    return gaussian_filter1d(traj, sigma=sigma)


# =============================================================================
# Phase 0: 全量数据加载与清洗
# =============================================================================


def phase0_load_full_data():
    """加载并清洗全量数据"""
    print("=" * 80)
    print("Phase 0: 全量数据清洗")
    print("=" * 80)

    all_sessions = []

    print("\n[Step 0.1] 加载深医全量数据...")
    shenyi_dir = Path("/home/cht/Works/OptiHemoAI/dataset/深医/透析数据")
    for year in [2018, 2019, 2020, 2021, 2022]:
        fp = shenyi_dir / f"{year}.xlsx"
        if not fp.exists():
            continue
        df = pd.read_excel(fp)
        print(f"  {year}: {len(df)} 条")
        for _, row in df.iterrows():
            s = parse_shenyi_full(row)
            if s:
                all_sessions.append(s)
    print(f"  深医有效: {len(all_sessions)} 条")

    print("\n[Step 0.2] 加载福鼎全量数据...")
    fuding_dir = Path(
        "/home/cht/Works/OptiHemoAI/dataset/2023-4-19 福鼎市医院HD数据-周鸿明整理后"
    )
    df_mid = pd.read_csv(fuding_dir / "福鼎透中20221025-透中信息用这个表(1).csv")
    df_pp = pd.read_csv(fuding_dir / "福鼎透前透后.csv")
    print(f"  透中: {len(df_mid)} 条, 透前透后: {len(df_pp)} 条")
    for _, row in df_mid.iterrows():
        s = parse_fuding_full(row, df_pp)
        if s:
            all_sessions.append(s)
    print(
        f"  福鼎有效: {len(all_sessions) - sum(1 for s in all_sessions if s['center']=='shenyi')} 条"
    )
    print(f"  总计: {len(all_sessions)} 条")

    print("\n[Step 0.3] 数据质量报告...")
    for center in ["shenyi", "fuding"]:
        cs = [s for s in all_sessions if s["center"] == center]
        sbp_vals = [x for s in cs for x in s["time_sbp"] if not np.isnan(x)]
        print(
            f"  {center.upper()}: n={len(cs)}, SBP范围={min(sbp_vals):.0f}-{max(sbp_vals):.0f}, 均值={np.mean(sbp_vals):.1f}±{np.std(sbp_vals):.1f}"
        )

    print("\n[Step 0.4] 保存全量清洗数据...")
    output = PROCESSED_DIR / "all_cleaned_full.pkl"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(output, "wb") as f:
        pickle.dump(all_sessions, f)
    print(f"  已保存: {output}")
    return all_sessions


# =============================================================================
# Phase 1: 两阶段聚类 (Optimized with Parallel DTW)
# =============================================================================


def euclidean_dist(x, y):
    return np.sqrt(np.sum((np.array(x) - np.array(y)) ** 2))


def fastdtw_distance(x, y):
    d, _ = fastdtw(x, y, dist=euclidean_dist)
    return d


def compute_dtw_distances_batch(X, centers):
    """Compute DTW distances for a batch of samples to all centers"""
    distances = np.zeros((len(X), len(centers)))
    for i, x in enumerate(X):
        for j, c in enumerate(centers):
            distances[i, j] = fastdtw_distance(x, c)
    return distances


def compute_dtw_distance_single(x, centers):
    """Compute DTW distances for a single sample to all centers"""
    distances = np.zeros(len(centers))
    for j, c in enumerate(centers):
        distances[j] = fastdtw_distance(x, c)
    return distances


def soft_dtw_knn_parallel(X, centers, n_jobs=N_JOBS):
    """Parallel DTW distance computation"""
    distances = Parallel(n_jobs=n_jobs)(
        delayed(compute_dtw_distance_single)(x, centers) for x in X
    )
    return np.array(distances)


def compute_centroids(X, labels):
    new_centers = []
    for k in range(N_CLUSTERS):
        cluster_pts = [X[i] for i in range(len(X)) if labels[i] == k]
        if not cluster_pts:
            new_centers.append(X[0])
            continue
        min_len = min(len(p) for p in cluster_pts)
        truncated = [p[:min_len] for p in cluster_pts]
        new_centers.append(np.mean(truncated, axis=0))
    return np.array(new_centers)


def soft_dtw_kmeans_step1(X_sample, max_iter=50):
    """步骤一: 在采样数据上运行DTW-KMeans,返回Centroids"""
    print(f"\n[Phase 1 - 步骤一] Soft-DTW K-Means on {len(X_sample)} samples...")
    print(f"  使用 {N_JOBS} 个并行核心")
    np.random.seed(RANDOM_STATE)
    idx = np.random.choice(len(X_sample), N_CLUSTERS, replace=False)
    centers = X_sample[idx].copy()
    labels = np.zeros(len(X_sample), dtype=int)

    for it in range(max_iter):
        print(f"  Iter {it}: 计算DTW距离矩阵...", flush=True)
        distances = soft_dtw_knn_parallel(X_sample, centers)
        new_labels = np.argmin(distances, axis=1)
        if np.array_equal(new_labels, labels):
            print(f"  收敛于 iteration {it}")
            break
        labels = new_labels
        centers = compute_centroids(X_sample, labels)
        print(f"  Iter {it}: 簇分布 {np.bincount(labels)}", flush=True)

    print(f"  最终Centroids形状: {centers.shape}")
    return labels, centers


def assign_remaining_to_centroids(X_remaining, centroids):
    """步骤三: 1-NN分配剩余数据到预设Centroids"""
    print(f"\n[Phase 1 - 步骤三] 1-NN分配 {len(X_remaining)} 条数据...", flush=True)
    distances = soft_dtw_knn_parallel(X_remaining, centroids)
    labels_remaining = np.argmin(distances, axis=1)
    print(f"  分配完成: {np.bincount(labels_remaining)}", flush=True)
    return labels_remaining


def phase1_two_stage_clustering(sessions):
    """两阶段聚类主流程"""
    print("\n" + "=" * 80)
    print("Phase 1: 两阶段聚类")
    print("=" * 80)

    print("\n[Phase 1.0] 提取ΔSBP轨迹...", flush=True)
    trajectories = []
    valid_sessions = []
    for s in sessions:
        traj = normalize_to_deltasbp(s)
        if traj is not None:
            trajectories.append(smooth_traj(traj, sigma=1))
            valid_sessions.append(s)

    X = np.array(trajectories)
    print(f"  有效轨迹: {len(X)}, 形状: {X.shape}", flush=True)

    print(f"\n[Phase 1.1] 分层采样 {SAMPLE_SIZE} 条...", flush=True)
    shenyi_idx = [i for i, s in enumerate(valid_sessions) if s["center"] == "shenyi"]
    fuding_idx = [i for i, s in enumerate(valid_sessions) if s["center"] == "fuding"]
    n_per = SAMPLE_SIZE // 2
    sample_idx = list(
        np.random.choice(shenyi_idx, min(n_per, len(shenyi_idx)), replace=False)
    )
    sample_idx += list(
        np.random.choice(fuding_idx, min(n_per, len(fuding_idx)), replace=False)
    )
    X_sample = X[sample_idx]
    sessions_sample = [valid_sessions[i] for i in sample_idx]
    remaining_idx = [i for i in range(len(X)) if i not in set(sample_idx)]
    X_remaining = X[remaining_idx]
    sessions_remaining = [valid_sessions[i] for i in remaining_idx]
    print(f"  采样: {len(X_sample)}, 剩余: {len(X_remaining)}", flush=True)

    labels_sample, centroids = soft_dtw_kmeans_step1(X_sample, max_iter=50)

    for i, s in enumerate(sessions_sample):
        s["phenotype_id"] = int(labels_sample[i])

    labels_remaining = assign_remaining_to_centroids(X_remaining, centroids)
    for i, s in enumerate(sessions_remaining):
        s["phenotype_id"] = int(labels_remaining[i])

    all_labeled = sessions_sample + sessions_remaining

    # 修复: 避免 O(n²) 的 index 查找，直接拼接轨迹数组
    X_labeled = np.vstack([X_sample, X_remaining])

    print(f"\n  全量聚类完成: {len(all_labeled)} 条", flush=True)
    for k in range(N_CLUSTERS):
        n_k = sum(1 for s in all_labeled if s["phenotype_id"] == k)
        print(f"  P{k}: {n_k} ({100*n_k/len(all_labeled):.1f}%)", flush=True)

    return all_labeled, X_labeled, centroids


def characterize_phenotype(centroid):
    final = centroid[-1]
    min_val = np.min(centroid)
    if final < -30 and min_val < -35:
        return "P0: Severe Drop (末端跳水型)"
    elif final > 5:
        return "P1: Rise (末端升高型)"
    elif final >= -15:
        return "P2: Stable (全程平稳型)"
    else:
        return "P3: Mild Drop (末端跳水型)"


def phase1_visualize(all_labeled, X_labeled, centroids):
    """生成4表型轨迹门面图"""
    print("\n[Phase 1] 生成表型轨迹图...", flush=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    colors = ["#c0392b", "#27ae60", "#2980b9", "#e67e22"]
    time_axis = np.linspace(0, 240, MAX_SEQ_LEN)

    for k in range(N_CLUSTERS):
        ax = axes[k]
        cluster_mask = np.array([s["phenotype_id"] == k for s in all_labeled])
        cluster_traj = X_labeled[cluster_mask]

        for traj in cluster_traj:
            ax.plot(time_axis, traj, color=colors[k], alpha=0.02, linewidth=0.5)

        mean_traj = np.mean(cluster_traj, axis=0)
        std_traj = np.std(cluster_traj, axis=0)
        n = len(cluster_traj)
        ci_lo = mean_traj - 1.96 * std_traj / np.sqrt(n)
        ci_hi = mean_traj + 1.96 * std_traj / np.sqrt(n)

        ax.plot(time_axis, mean_traj, color=colors[k], linewidth=3)
        ax.fill_between(time_axis, ci_lo, ci_hi, color=colors[k], alpha=0.3)
        ax.plot(
            time_axis,
            centroids[k],
            color="black",
            linewidth=2,
            linestyle="--",
            label="Centroid",
        )
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.axhline(-20, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("ΔSBP (mmHg)")
        phenotype_name = characterize_phenotype(centroids[k])
        n_k = sum(1 for s in all_labeled if s["phenotype_id"] == k)
        ax.set_title(
            f"{phenotype_name}\nn={n_k} ({100*n_k/len(all_labeled):.1f}%)", fontsize=11
        )
        ax.legend(loc="best")
        ax.set_xlim(0, 240)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Project Landscape V2: 30万全量数据血流动力学表型图鉴\n(Soft-DTW K-Means, K=4)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "Figure1_Phenotype_Landscape_Full.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()
    print(
        f"  已保存: {FIGURES_DIR / 'Figure1_Phenotype_Landscape_Full.png'}", flush=True
    )


# =============================================================================
# Phase 2: Root Cause Analysis
# =============================================================================


def phase2_root_cause(all_labeled, phenotype_stats):
    """追查福鼎P0高发的真凶"""
    print("\n" + "=" * 80)
    print("Phase 2: Root Cause Analysis (UFR & IDWG)")
    print("=" * 80)

    results = {"by_phenotype": {}, "by_center": {}}

    print("\n[Phase 2.1] 各表型UFR/IDWG统计...", flush=True)
    for k in range(N_CLUSTERS):
        cluster = [s for s in all_labeled if s["phenotype_id"] == k]
        ufr_vals = [
            s["prescribed_ufr"]
            for s in cluster
            if not np.isnan(s.get("prescribed_ufr", np.nan))
        ]
        idwg_vals = [s["idwg"] for s in cluster if not np.isnan(s.get("idwg", np.nan))]
        ufr_mean = np.mean(ufr_vals) if ufr_vals else np.nan
        idwg_mean = np.mean(idwg_vals) if idwg_vals else np.nan
        results["by_phenotype"][k] = {
            "n": len(cluster),
            "ufr_mean": ufr_mean,
            "ufr_std": np.std(ufr_vals) if ufr_vals else np.nan,
            "idwg_mean": idwg_mean,
            "idwg_std": np.std(idwg_vals) if idwg_vals else np.nan,
        }
        ufr_label = f"{ufr_mean:.1f}±{np.std(ufr_vals):.1f}" if ufr_vals else "N/A"
        idwg_label = f"{idwg_mean:.2f}±{np.std(idwg_vals):.2f}" if idwg_vals else "N/A"
        print(
            f"  P{k} ({characterize_phenotype(phenotype_stats[k]['centroid'])[:20]}): UFR={ufr_label} ml/hr, IDWG={idwg_label} kg",
            flush=True,
        )

    print("\n[Phase 2.2] 福鼎 vs 深医 UFR/IDWG 差异检验...", flush=True)
    shenyi_sessions = [s for s in all_labeled if s["center"] == "shenyi"]
    fuding_sessions = [s for s in all_labeled if s["center"] == "fuding"]

    shenyi_ufr = [
        s["prescribed_ufr"]
        for s in shenyi_sessions
        if not np.isnan(s.get("prescribed_ufr", np.nan))
    ]
    fuding_ufr = [
        s["prescribed_ufr"]
        for s in fuding_sessions
        if not np.isnan(s.get("prescribed_ufr", np.nan))
    ]
    shenyi_idwg = [
        s["idwg"] for s in shenyi_sessions if not np.isnan(s.get("idwg", np.nan))
    ]
    fuding_idwg = [
        s["idwg"] for s in fuding_sessions if not np.isnan(s.get("idwg", np.nan))
    ]

    t_ufr, p_ufr = ttest_ind(shenyi_ufr, fuding_ufr, nan_policy="omit")
    t_idwg, p_idwg = ttest_ind(shenyi_idwg, fuding_idwg, nan_policy="omit")

    print(f"\n  【处方超滤率 (Prescribed UFR)】", flush=True)
    print(
        f"    深医: {np.mean(shenyi_ufr):.1f} ± {np.std(shenyi_ufr):.1f} ml/hr (n={len(shenyi_ufr)})",
        flush=True,
    )
    print(
        f"    福鼎: {np.mean(fuding_ufr):.1f} ± {np.std(fuding_ufr):.1f} ml/hr (n={len(fuding_ufr)})",
        flush=True,
    )
    print(
        f"    差异检验: t={t_ufr:.3f}, p={p_ufr:.2e} {'***' if p_ufr<0.001 else '**' if p_ufr<0.01 else '*' if p_ufr<0.05 else ''}",
        flush=True,
    )

    print(f"\n  【透间体重增长 (IDWG)】", flush=True)
    print(
        f"    深医: {np.mean(shenyi_idwg):.2f} ± {np.std(shenyi_idwg):.2f} kg (n={len(shenyi_idwg)})",
        flush=True,
    )
    print(
        f"    福鼎: {np.mean(fuding_idwg):.2f} ± {np.std(fuding_idwg):.2f} kg (n={len(fuding_idwg)})",
        flush=True,
    )
    print(
        f"    差异检验: t={t_idwg:.3f}, p={p_idwg:.2e} {'***' if p_idwg<0.001 else '**' if p_idwg<0.01 else '*' if p_idwg<0.05 else ''}",
        flush=True,
    )

    print(f"\n  【福鼎P0泛滥真凶分析】", flush=True)
    p0_ufr = results["by_phenotype"][0]["ufr_mean"]
    other_ufr = np.mean(
        [
            results["by_phenotype"][k]["ufr_mean"]
            for k in [1, 2, 3]
            if not np.isnan(results["by_phenotype"][k]["ufr_mean"])
        ]
    )
    print(
        f"    P0(UFR={p0_ufr:.1f}) vs 其他表型平均(UFR={other_ufr:.1f}): 比值={p0_ufr/other_ufr:.2f}x",
        flush=True,
    )

    fuding_high_ufr = (
        sum(1 for u in fuding_ufr if u > np.median(fuding_ufr)) / len(fuding_ufr) * 100
    )
    shenyi_high_ufr = (
        sum(1 for u in shenyi_ufr if u > np.median(shenyi_ufr)) / len(shenyi_ufr) * 100
    )
    print(
        f"    福鼎高UFR(>中位数)占比: {fuding_high_ufr:.1f}% vs 深医: {shenyi_high_ufr:.1f}%",
        flush=True,
    )

    results["center_comparison"] = {
        "shenyi_ufr": (np.mean(shenyi_ufr), np.std(shenyi_ufr), len(shenyi_ufr)),
        "fuding_ufr": (np.mean(fuding_ufr), np.std(fuding_ufr), len(fuding_ufr)),
        "t_ufr": t_ufr,
        "p_ufr": p_ufr,
        "shenyi_idwg": (np.mean(shenyi_idwg), np.std(shenyi_idwg), len(shenyi_idwg)),
        "fuding_idwg": (np.mean(fuding_idwg), np.std(fuding_idwg), len(fuding_idwg)),
        "t_idwg": t_idwg,
        "p_idwg": p_idwg,
    }
    return results


def phase2_plot(results):
    """生成UFR/IDWG对比图"""
    print("\n[Phase 2.3] 生成Root Cause图表...", flush=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    phenotype_ufr = [results["by_phenotype"][k]["ufr_mean"] for k in range(N_CLUSTERS)]
    phenotype_ufr_std = [
        results["by_phenotype"][k]["ufr_std"] for k in range(N_CLUSTERS)
    ]
    colors_bar = ["#c0392b", "#27ae60", "#2980b9", "#e67e22"]

    x = np.arange(N_CLUSTERS)
    bars = axes[0, 0].bar(
        x, phenotype_ufr, yerr=phenotype_ufr_std, capsize=4, color=colors_bar, alpha=0.8
    )
    axes[0, 0].set_xlabel("Phenotype")
    axes[0, 0].set_ylabel("Prescribed UFR (ml/hr)")
    axes[0, 0].set_title("各表型处方超滤率\n(Prescribed UFR = 超滤量/透析时长)")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels([f"P{k}" for k in range(N_CLUSTERS)])
    for bar, val in zip(bars, phenotype_ufr):
        axes[0, 0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 20,
            f"{val:.0f}",
            ha="center",
            fontsize=9,
        )
    axes[0, 0].grid(True, alpha=0.3, axis="y")

    center_ufr_means = [
        results["center_comparison"]["shenyi_ufr"][0],
        results["center_comparison"]["fuding_ufr"][0],
    ]
    center_ufr_stds = [
        results["center_comparison"]["shenyi_ufr"][1],
        results["center_comparison"]["fuding_ufr"][1],
    ]
    p_ufr = results["center_comparison"]["p_ufr"]
    bars2 = axes[0, 1].bar(
        ["深医", "福鼎"],
        center_ufr_means,
        yerr=center_ufr_stds,
        capsize=5,
        color=["#3498db", "#e74c3c"],
        alpha=0.8,
    )
    axes[0, 1].set_ylabel("Prescribed UFR (ml/hr)")
    axes[0, 1].set_title(f"深医 vs 福鼎 处方超滤率\n(p={p_ufr:.2e})")
    for bar, val in zip(bars2, center_ufr_means):
        axes[0, 1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 30,
            f"{val:.0f}",
            ha="center",
            fontsize=10,
        )
    axes[0, 1].grid(True, alpha=0.3, axis="y")

    phenotype_idwg = [
        results["by_phenotype"][k]["idwg_mean"] for k in range(N_CLUSTERS)
    ]
    phenotype_idwg_std = [
        results["by_phenotype"][k]["idwg_std"] for k in range(N_CLUSTERS)
    ]
    bars3 = axes[1, 0].bar(
        x,
        phenotype_idwg,
        yerr=phenotype_idwg_std,
        capsize=4,
        color=colors_bar,
        alpha=0.8,
    )
    axes[1, 0].set_xlabel("Phenotype")
    axes[1, 0].set_ylabel("IDWG (kg)")
    axes[1, 0].set_title("各表型透间体重增长\n(IDWG = 透前体重 - 干体重)")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels([f"P{k}" for k in range(N_CLUSTERS)])
    for bar, val in zip(bars3, phenotype_idwg):
        axes[1, 0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{val:.2f}",
            ha="center",
            fontsize=9,
        )
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    center_idwg_means = [
        results["center_comparison"]["shenyi_idwg"][0],
        results["center_comparison"]["fuding_idwg"][0],
    ]
    center_idwg_stds = [
        results["center_comparison"]["shenyi_idwg"][1],
        results["center_comparison"]["fuding_idwg"][1],
    ]
    p_idwg = results["center_comparison"]["p_idwg"]
    bars4 = axes[1, 1].bar(
        ["深医", "福鼎"],
        center_idwg_means,
        yerr=center_idwg_stds,
        capsize=5,
        color=["#3498db", "#e74c3c"],
        alpha=0.8,
    )
    axes[1, 1].set_ylabel("IDWG (kg)")
    axes[1, 1].set_title(f"深医 vs 福鼎 透间体重增长\n(p={p_idwg:.2e})")
    for bar, val in zip(bars4, center_idwg_means):
        axes[1, 1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{val:.2f}",
            ha="center",
            fontsize=10,
        )
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        "Project Landscape V2: 福鼎P0高发 Root Cause Analysis",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "Figure2_Root_Cause_Analysis.png", dpi=200, bbox_inches="tight"
    )
    plt.close()
    print(f"  已保存: {FIGURES_DIR / 'Figure2_Root_Cause_Analysis.png'}", flush=True)


# =============================================================================
# Phase 3: LightGBM + SHAP
# =============================================================================


def phase3_predict_with_shap(all_labeled):
    """分中心建模: 深医和福鼎分别训练 LightGBM + SHAP解释性"""
    print("\n" + "=" * 80)
    print("Phase 3: 分中心表型预测 + SHAP解释性 (UFR语义增强版)")
    print("=" * 80)

    feature_names_base = [
        "年龄",
        "透前体重",
        "干体重",
        "透前SBP",
        "透前DBP",
        "处方UFR (ml/hr)",
        "IDWG (kg)",
        "性别(男=1)",
    ]

    results_by_center = {}

    for center, center_name in [("shenyi", "深医"), ("fuding", "福鼎")]:
        print(f"\n{'='*60}")
        print(f"  {center_name}中心模型")
        print(f"{'='*60}")

        features, labels = [], []
        for s in all_labeled:
            if s["center"] != center:
                continue

            prescribed = s.get("prescribed_ufr", np.nan)

            # 福鼎 pre_sbp/pre_dbp 100% 缺失，从 time_sbp[0]/time_dbp[0] 反推
            pre_sbp = s.get("pre_sbp", np.nan)
            if np.isnan(pre_sbp):
                ts = s.get("time_sbp", [])
                if ts and len(ts) > 0 and ts[0] > 0 and not np.isnan(ts[0]):
                    pre_sbp = ts[0]

            pre_dbp = s.get("pre_dbp", np.nan)
            if np.isnan(pre_dbp):
                td = s.get("time_dbp", [])
                if td and len(td) > 0 and td[0] > 0 and not np.isnan(td[0]):
                    pre_dbp = td[0]

            feat = [
                s.get("age", np.nan),
                s.get("pre_weight", np.nan),
                s.get("dry_weight", np.nan),
                pre_sbp,
                pre_dbp,
                prescribed,
                s.get("idwg", np.nan),
                1.0 if s.get("sex") == "男" else 0.0,
            ]
            features.append(feat)
            labels.append(s["phenotype_id"])

        X_feat = np.array(features)
        y = np.array(labels)

        print(f"  总样本: {len(y)}", flush=True)

        # 使用 KNN 插补处理缺失值
        from sklearn.impute import KNNImputer

        imputer = KNNImputer(n_neighbors=5)
        X_feat_imputed = imputer.fit_transform(X_feat)

        valid_mask = ~np.isnan(X_feat).any(axis=1)
        n_valid = valid_mask.sum()
        n_imputed = (~valid_mask).sum()
        print(f"  完整样本: {n_valid}, 插补样本: {n_imputed}", flush=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X_feat_imputed, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )
        print(f"  训练: {len(y_train)}, 测试: {len(y_test)}", flush=True)

        model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            random_state=RANDOM_STATE,
            verbose=-1,
            class_weight="balanced",
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)])

        y_pred_proba = model.predict_proba(X_test)
        y_pred = model.predict(X_test)
        accuracy = np.mean(y_pred == y_test)
        print(f"\n  整体准确率: {accuracy:.3f}", flush=True)

        print("\n  各表型AUROC:", flush=True)
        auc_scores = {}
        n_classes = N_CLUSTERS
        for k in range(n_classes):
            y_binary = (y_test == k).astype(int)
            if len(np.unique(y_binary)) == 2:
                auc = roc_auc_score(y_binary, y_pred_proba[:, k])
                auc_scores[k] = auc
                print(f"    P{k}: AUROC={auc:.3f}", flush=True)

        print(f"\n  P0 (Severe Drop) 单独预测:", flush=True)
        y_binary_p0 = (y_test == 0).astype(int)
        auroc_p0 = roc_auc_score(y_binary_p0, y_pred_proba[:, 0])
        ap_p0 = average_precision_score(y_binary_p0, y_pred_proba[:, 0])
        print(f"    AUROC={auroc_p0:.3f}, AUPRC={ap_p0:.3f}", flush=True)

        # SHAP 分析
        print("\n  计算SHAP值 (P0专属)...", flush=True)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        print(f"  shap_values 形状: {shap_values.shape}", flush=True)
        sv_p0 = shap_values[:, :, 0]
        print(f"  P0 SHAP值形状: {sv_p0.shape}", flush=True)

        results_by_center[center] = {
            "accuracy": accuracy,
            "auc_scores": auc_scores,
            "auroc_p0": auroc_p0,
            "ap_p0": ap_p0,
            "shap_values": sv_p0,
            "feature_names": feature_names_base,
            "model": model,
            "X_test": X_test,
            "y_test": y_test,
            "y_pred_proba": y_pred_proba,
            "y_pred": y_pred,
        }

    # 生成分中心对比图
    print("\n" + "=" * 60)
    print("生成分中心对比图表")
    print("=" * 60)

    # Figure 3A: 双中心 AUROC 对比
    fig, ax = plt.subplots(figsize=(10, 8))
    colors_roc = ["#c0392b", "#27ae60", "#2980b9", "#e67e22"]

    for center, center_name, alpha in [
        ("shenyi", "深医", 1.0),
        ("fuding", "福鼎", 0.6),
    ]:
        res = results_by_center[center]
        for k in range(N_CLUSTERS):
            y_binary = (res["y_test"] == k).astype(int)
            if len(np.unique(y_binary)) == 2:
                fpr, tpr, _ = roc_curve(y_binary, res["y_pred_proba"][:, k])
                label = f"P{k} ({center_name})"
                ax.plot(
                    fpr, tpr, color=colors_roc[k], linewidth=2, alpha=alpha, label=label
                )

    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("FPR", fontsize=12)
    ax.set_ylabel("TPR", fontsize=12)
    ax.set_title(
        "Multi-class ROC Curves (Shenyi vs Fuding)", fontsize=13, fontweight="bold"
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "Figure3_ROC_Comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.close()
    print(f"  已保存: {FIGURES_DIR / 'Figure3_ROC_Comparison.png'}", flush=True)

    # Figure 3B: 双中心 SHAP Feature Importance 对比
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    for idx, (center, center_name) in enumerate(
        [("shenyi", "深医"), ("fuding", "福鼎")]
    ):
        res = results_by_center[center]
        sv_p0 = res["shap_values"]
        feature_names = res["feature_names"]

        shap_mean = np.abs(sv_p0).mean(axis=0)
        sorted_idx = np.argsort(shap_mean)
        colors_shap = [
            "#2ecc71" if "UFR" in f else "#3498db"
            for f in np.array(feature_names)[sorted_idx]
        ]
        bars = axes[idx].barh(
            np.array(feature_names)[sorted_idx],
            shap_mean[sorted_idx],
            color=colors_shap,
        )
        axes[idx].set_xlabel("Mean |SHAP value|", fontsize=11)
        axes[idx].set_title(
            f"SHAP Feature Importance (P0) - {center_name}",
            fontsize=12,
            fontweight="bold",
        )
        axes[idx].grid(axis="x", alpha=0.3)

        for bar, val in zip(bars, shap_mean[sorted_idx]):
            axes[idx].text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}",
                va="center",
                fontsize=9,
            )

    plt.suptitle(
        "Project Landscape V2: 分中心 SHAP 特征重要性对比 (P0)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "Figure3_SHAP_Comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.close()
    print(f"  已保存: {FIGURES_DIR / 'Figure3_SHAP_Comparison.png'}", flush=True)

    # Figure 3C: 深医 SHAP Summary Plot
    plt.figure(figsize=(10, 8))
    res_shenyi = results_by_center["shenyi"]
    shap.summary_plot(
        res_shenyi["shap_values"],
        res_shenyi["X_test"],
        feature_names=res_shenyi["feature_names"],
        show=False,
        max_display=10,
    )
    plt.title(
        "SHAP Summary Plot: P0 (深医)\nHigher SHAP → Higher P0 Risk",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "Figure3_SHAP_Summary_P0.png", dpi=300, bbox_inches="tight"
    )
    plt.close()
    print(f"  已保存: {FIGURES_DIR / 'Figure3_SHAP_Summary_P0.png'}", flush=True)

    # Figure 3D: 福鼎 SHAP Summary Plot
    plt.figure(figsize=(10, 8))
    res_fuding = results_by_center["fuding"]
    shap.summary_plot(
        res_fuding["shap_values"],
        res_fuding["X_test"],
        feature_names=res_fuding["feature_names"],
        show=False,
        max_display=10,
    )
    plt.title(
        "SHAP Summary Plot: P0 (福鼎)\nHigher SHAP → Higher P0 Risk",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "Figure3_SHAP_Summary_P0_Fuding.png", dpi=300, bbox_inches="tight"
    )
    plt.close()
    print(f"  已保存: {FIGURES_DIR / 'Figure3_SHAP_Summary_P0_Fuding.png'}", flush=True)

    # Figure 3E: UFR 特征专项分析 (双中心对比)
    fig_ufr, ax_ufr = plt.subplots(figsize=(12, 8))

    for idx, (center, center_name, color) in enumerate(
        [("shenyi", "深医", "#3498db"), ("fuding", "福鼎", "#e74c3c")]
    ):
        res = results_by_center[center]
        sv_p0 = res["shap_values"]
        feature_names = res["feature_names"]

        ufr_feat_idx = [i for i, f in enumerate(feature_names) if "UFR" in f]
        ufr_shap_vals = sv_p0[:, ufr_feat_idx]
        ufr_names = [feature_names[i] for i in ufr_feat_idx]

        for i, (name, vals) in enumerate(zip(ufr_names, ufr_shap_vals.T)):
            pos = i + idx * 0.3
            violin = ax_ufr.violinplot(
                vals, positions=[pos], widths=0.25, showmeans=True, showmedians=True
            )
            for pc in violin["bodies"]:
                pc.set_facecolor(color)
                pc.set_alpha(0.6)

    ax_ufr.set_xticks([i + 0.15 for i in range(len(ufr_names))])
    ax_ufr.set_xticklabels(ufr_names, fontsize=11)
    ax_ufr.set_ylabel("SHAP Value (P0 Risk Contribution)", fontsize=12)
    ax_ufr.set_title(
        "UFR Feature SHAP Distribution (P0) - 双中心对比\nBlue=深医, Red=福鼎",
        fontsize=13,
        fontweight="bold",
    )
    ax_ufr.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax_ufr.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "Figure3_SHAP_UFR_Special.png", dpi=300, bbox_inches="tight"
    )
    plt.close()
    print(f"  已保存: {FIGURES_DIR / 'Figure3_SHAP_UFR_Special.png'}", flush=True)

    # Figure 3F: 预测模型总览 (双中心对比)
    fig_overview, axes_overview = plt.subplots(2, 2, figsize=(16, 12))

    # 准确率对比
    centers = ["深医", "福鼎"]
    accuracies = [
        results_by_center["shenyi"]["accuracy"],
        results_by_center["fuding"]["accuracy"],
    ]
    bars_acc = axes_overview[0, 0].bar(
        centers, accuracies, color=["#3498db", "#e74c3c"], alpha=0.8
    )
    axes_overview[0, 0].set_ylabel("Accuracy", fontsize=12)
    axes_overview[0, 0].set_title(
        "Overall Prediction Accuracy", fontsize=12, fontweight="bold"
    )
    axes_overview[0, 0].set_ylim(0, 1)
    for bar, val in zip(bars_acc, accuracies):
        axes_overview[0, 0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center",
            fontsize=11,
        )
    axes_overview[0, 0].grid(axis="y", alpha=0.3)

    # P0 AUROC 对比
    aurocs_p0 = [
        results_by_center["shenyi"]["auroc_p0"],
        results_by_center["fuding"]["auroc_p0"],
    ]
    bars_auroc = axes_overview[0, 1].bar(
        centers, aurocs_p0, color=["#3498db", "#e74c3c"], alpha=0.8
    )
    axes_overview[0, 1].set_ylabel("AUROC (P0)", fontsize=12)
    axes_overview[0, 1].set_title("P0 Prediction AUROC", fontsize=12, fontweight="bold")
    axes_overview[0, 1].set_ylim(0, 1)
    for bar, val in zip(bars_auroc, aurocs_p0):
        axes_overview[0, 1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center",
            fontsize=11,
        )
    axes_overview[0, 1].grid(axis="y", alpha=0.3)

    # 深医混淆矩阵
    res_s = results_by_center["shenyi"]
    conf_s = np.zeros((N_CLUSTERS, N_CLUSTERS))
    for true, pred in zip(res_s["y_test"], res_s["y_pred"]):
        conf_s[true, pred] += 1
    im_s = axes_overview[1, 0].imshow(conf_s, cmap="Blues")
    axes_overview[1, 0].set_xticks(range(N_CLUSTERS))
    axes_overview[1, 0].set_yticks(range(N_CLUSTERS))
    axes_overview[1, 0].set_xticklabels([f"P{k}" for k in range(N_CLUSTERS)])
    axes_overview[1, 0].set_yticklabels([f"P{k}" for k in range(N_CLUSTERS)])
    axes_overview[1, 0].set_xlabel("Predicted")
    axes_overview[1, 0].set_ylabel("True")
    axes_overview[1, 0].set_title("Confusion Matrix (深医)")
    for i in range(N_CLUSTERS):
        for j in range(N_CLUSTERS):
            axes_overview[1, 0].text(
                j,
                i,
                int(conf_s[i, j]),
                ha="center",
                va="center",
                color="white" if conf_s[i, j] > conf_s.max() / 2 else "black",
            )
    plt.colorbar(im_s, ax=axes_overview[1, 0])

    # 福鼎混淆矩阵
    res_f = results_by_center["fuding"]
    conf_f = np.zeros((N_CLUSTERS, N_CLUSTERS))
    for true, pred in zip(res_f["y_test"], res_f["y_pred"]):
        conf_f[true, pred] += 1
    im_f = axes_overview[1, 1].imshow(conf_f, cmap="Reds")
    axes_overview[1, 1].set_xticks(range(N_CLUSTERS))
    axes_overview[1, 1].set_yticks(range(N_CLUSTERS))
    axes_overview[1, 1].set_xticklabels([f"P{k}" for k in range(N_CLUSTERS)])
    axes_overview[1, 1].set_yticklabels([f"P{k}" for k in range(N_CLUSTERS)])
    axes_overview[1, 1].set_xlabel("Predicted")
    axes_overview[1, 1].set_ylabel("True")
    axes_overview[1, 1].set_title("Confusion Matrix (福鼎)")
    for i in range(N_CLUSTERS):
        for j in range(N_CLUSTERS):
            axes_overview[1, 1].text(
                j,
                i,
                int(conf_f[i, j]),
                ha="center",
                va="center",
                color="white" if conf_f[i, j] > conf_f.max() / 2 else "black",
            )
    plt.colorbar(im_f, ax=axes_overview[1, 1])

    plt.suptitle(
        "Project Landscape V2: 分中心预测模型对比\n(LightGBM, 深医 vs 福鼎)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "Figure3_Phenotype_Prediction_Full.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print(
        f"  已保存: {FIGURES_DIR / 'Figure3_Phenotype_Prediction_Full.png'}", flush=True
    )

    return results_by_center


# =============================================================================
# 主函数
# =============================================================================


def main():
    print("=" * 80)
    print("Project Landscape V2: 30万全量数据 + 顶刊发表图表")
    print("=" * 80)

    cache_path = PROCESSED_DIR / "all_cleaned_full_v2.pkl"
    if cache_path.exists():
        print("\n[缓存命中] 加载已清洗全量数据 (V2 with corrected UFR)...")
        with open(cache_path, "rb") as f:
            all_sessions = pickle.load(f)
        print(f"  总计: {len(all_sessions)} 条")

        # Verify UFR fields
        ufr_vals = [
            s["prescribed_ufr"]
            for s in all_sessions
            if not np.isnan(s.get("prescribed_ufr", np.nan))
        ]
        print(
            f"  prescribed_ufr: {len(ufr_vals)} values, mean={np.mean(ufr_vals):.0f} ml/hr"
        )
    else:
        print(f"\n[警告] 未找到V2缓存: {cache_path}")
        print("请先运行 add_ufr_fields.py 生成 all_cleaned_full_v2.pkl")
        return

    all_labeled, X_labeled, centroids = phase1_two_stage_clustering(all_sessions)

    phenotype_stats = {}
    for k in range(N_CLUSTERS):
        cluster_mask = np.array([s["phenotype_id"] == k for s in all_labeled])
        cluster_X = X_labeled[cluster_mask]
        phenotype_stats[k] = {
            "centroid": centroids[k],
            "n": int(np.sum(cluster_mask)),
            "name": characterize_phenotype(centroids[k]),
        }

    phase1_visualize(all_labeled, X_labeled, centroids)

    for s in all_labeled:
        s["idh_label"] = extract_idh_label(s)

    adverse = {}
    for k in range(N_CLUSTERS):
        cluster = [s for s in all_labeled if s["phenotype_id"] == k]
        n_idh = sum(1 for s in cluster if s["idh_label"] == "IDH")
        adverse[k] = {
            "n": len(cluster),
            "idh_rate": n_idh / len(cluster) * 100 if cluster else 0,
        }
        print(f"  P{k} IDH率: {adverse[k]['idh_rate']:.1f}%")

    root_cause_results = phase2_root_cause(all_labeled, phenotype_stats)
    phase2_plot(root_cause_results)

    shap_results = phase3_predict_with_shap(all_labeled)

    print("\n" + "=" * 80)
    print("Project Landscape V2 完成!")
    print("=" * 80)
    print(
        f"""
核心发现:
1. 全量30万数据聚类完成 (两阶段DTW-KMeans)
2. Root Cause: 福鼎UFR={root_cause_results['center_comparison']['fuding_ufr'][0]:.0f} vs 深医={root_cause_results['center_comparison']['shenyi_ufr'][0]:.0f} ml/hr (p={root_cause_results['center_comparison']['p_ufr']:.2e})
3. P0专属SHAP解释性图已生成 (UFR方向性权重)
4. 分中心预测AUROC(P0):
   - 深医: {shap_results['shenyi']['auroc_p0']:.3f}
   - 福鼎: {shap_results['fuding']['auroc_p0']:.3f}

输出文件:
- Figure1_Phenotype_Landscape_Full.png  (表型轨迹图)
- Figure2_Root_Cause_Analysis.png       (UFR/IDWG分析)
- Figure3_ROC_Comparison.png           (双中心AUROC对比)
- Figure3_SHAP_Comparison.png          (双中心SHAP特征重要性对比)
- Figure3_SHAP_Summary_P0.png          (深医SHAP解释性)
- Figure3_SHAP_Summary_P0_Fuding.png   (福鼎SHAP解释性)
- Figure3_SHAP_UFR_Special.png         (UFR特征专项分析)
- Figure3_Phenotype_Prediction_Full.png (预测模型总览)
"""
    )


if __name__ == "__main__":
    main()
