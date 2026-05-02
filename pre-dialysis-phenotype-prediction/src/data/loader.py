"""Data loading and feature extraction for phenotype prediction."""

import ast
import re
import numpy as np
import pandas as pd


def compute_ufr_fields(row, center):
    """Compute UFR-related fields with center-specific semantics."""
    result = {"prescribed_ufr": np.nan, "actual_ufr": np.nan}
    try:
        duration = float(row.get("实际透析时长", np.nan))
        if duration <= 0 or np.isnan(duration):
            return result
    except (ValueError, TypeError):
        return result

    if center == "shenyi":
        try:
            prescribed_ufv = float(row.get("透析前预设UFV", np.nan))
            if prescribed_ufv > 0:
                result["prescribed_ufr"] = (prescribed_ufv * 1000) / duration
        except (ValueError, TypeError):
            pass
    elif center == "fuding":
        try:
            ufr_l_hr = float(row.get("UFR", np.nan))
            if ufr_l_hr > 0:
                result["prescribed_ufr"] = ufr_l_hr * 1000
        except (ValueError, TypeError):
            pass
    return result


def compute_idwg(row):
    """Calculate interdialytic weight gain: pre_weight - dry_weight (kg)."""
    try:
        pre_weight = float(row.get("透前体重", np.nan))
        dry_weight = float(row.get("干体重", np.nan))
        if pre_weight > 0 and dry_weight > 0:
            return pre_weight - dry_weight
    except (ValueError, TypeError):
        pass
    return np.nan


def validate_bp(sbp, dbp=None):
    """Validate blood pressure values."""
    if sbp is None or np.isnan(sbp) or sbp < 40 or sbp > 300:
        return False
    if dbp is not None and not np.isnan(dbp):
        if dbp < 20 or dbp > 150 or sbp <= dbp:
            return False
    return True


def parse_shenyi_features(row):
    """Parse Shenyi center record for feature extraction."""
    try:
        ufr_fields = compute_ufr_fields(row, "shenyi")
        pre_bp = str(row.get("透前血压", ""))
        pre_sbp, pre_dbp = np.nan, np.nan
        if "/" in pre_bp:
            parts = pre_bp.split("/")
            pre_sbp = float(parts[0]) if parts[0] else np.nan
            pre_dbp = float(parts[1]) if len(parts) > 1 and parts[1] else np.nan

        if not validate_bp(pre_sbp, pre_dbp):
            return None

        features = {
            "center": "shenyi",
            "patient_id": str(row["患者id"]),
            "session_id": str(row["透析记录id"]),
            "dialysis_date": row["透析日期"],
            "age": row.get("年龄", np.nan),
            "sex": 1 if row.get("性别", "未知") == "男" else 0,
            "pre_sbp": pre_sbp,
            "pre_dbp": pre_dbp,
            "prescribed_ufr": ufr_fields["prescribed_ufr"],
            "idwg": compute_idwg(row),
            "dialysis_duration": row.get("实际透析时长", 4.0),
            "dry_weight": row.get("干体重", np.nan),
            "pre_weight": row.get("透前体重", np.nan),
        }
        return features
    except Exception:
        return None


def parse_fuding_features(row, pre_post_df):
    """Parse Fuding center record for feature extraction."""
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

        ufr_fields = compute_ufr_fields(row, "fuding")
        pre_sbp_val = pp_row.get("透前收缩压", np.nan)
        pre_dbp_val = pp_row.get("透前舒张压", np.nan)
        pre_sbp = float(pre_sbp_val) if not pd.isna(pre_sbp_val) else np.nan
        pre_dbp = float(pre_dbp_val) if not pd.isna(pre_dbp_val) else np.nan

        if not validate_bp(pre_sbp, pre_dbp):
            return None

        features = {
            "center": "fuding",
            "patient_id": patient_name,
            "session_id": f"{patient_name}_{dialysis_date_raw}",
            "dialysis_date": dialysis_date_raw,
            "age": row.get("AGE", np.nan),
            "sex": 1,
            "pre_sbp": pre_sbp,
            "pre_dbp": pre_dbp,
            "prescribed_ufr": ufr_fields["prescribed_ufr"],
            "idwg": compute_idwg(row),
            "dialysis_duration": float(row.get("实际透析时长", 4.0)),
            "dry_weight": pp_row.get("干体重", np.nan),
            "pre_weight": pp_row.get("透前体重", np.nan),
        }
        return features
    except Exception:
        return None


def extract_features(all_sessions, phenotype_labels=None):
    """Extract feature matrix and labels from parsed sessions."""
    from config.settings import FEATURE_NAMES_BASE

    feature_names = FEATURE_NAMES_BASE.copy()
    X = []
    y = []

    for s in all_sessions:
        row = [s.get(f, np.nan) for f in feature_names]
        X.append(row)
        if phenotype_labels is not None:
            y.append(phenotype_labels.get(s["session_id"], -1))
        else:
            y.append(-1)

    return np.array(X), np.array(y), feature_names
