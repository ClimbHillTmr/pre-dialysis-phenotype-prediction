"""Data loading and preprocessing for hemodynamic phenotype analysis."""

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
        try:
            uf_volume_str = str(row.get("超滤量", ""))
            if uf_volume_str and uf_volume_str != "nan":
                uf_list = ast.literal_eval(uf_volume_str)
                if isinstance(uf_list, list):
                    for val in reversed(uf_list):
                        if val != "NA" and val is not None:
                            uf_volume = float(val)
                            if uf_volume > 0:
                                result["actual_ufr"] = uf_volume / duration
                            break
        except (ValueError, TypeError, SyntaxError):
            pass
    elif center == "fuding":
        try:
            ufr_l_hr = float(row.get("UFR", np.nan))
            if ufr_l_hr > 0:
                result["prescribed_ufr"] = ufr_l_hr * 1000
                result["actual_ufr"] = ufr_l_hr * 1000
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


def convert_to_float_list(input_list):
    """Convert string or list to float list, handling NA values."""
    if isinstance(input_list, str):
        try:
            parsed_list = ast.literal_eval(input_list)
            if isinstance(parsed_list, list):
                return [
                    float(v)
                    for v in parsed_list
                    if isinstance(v, (int, float, str)) and v and v != "NA"
                ]
        except (ValueError, TypeError, SyntaxError):
            pass
    elif isinstance(input_list, list):
        return [
            float(v)
            for v in input_list
            if isinstance(v, (int, float, str)) and v and v != "NA"
        ]
    return []


def validate_bp(sbp, dbp=None):
    """Validate blood pressure values."""
    if sbp is None or np.isnan(sbp) or sbp < 40 or sbp > 300:
        return False
    if dbp is not None and not np.isnan(dbp):
        if dbp < 20 or dbp > 150 or sbp <= dbp:
            return False
    return True


def validate_hr(hr):
    """Validate heart rate values."""
    if hr is None or np.isnan(hr):
        return True
    return 30 <= hr <= 250


def validate_ufr(ufr):
    """Validate ultrafiltration rate values."""
    if ufr is None or np.isnan(ufr):
        return True
    return 0 <= ufr <= 2000


def parse_shenyi(row):
    """Parse Shenyi center record."""
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
    except Exception:
        return None


def parse_fuding(row, pre_post_df):
    """Parse Fuding center record."""
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
        session = {
            "center": "fuding",
            "patient_id": patient_name,
            "session_id": f"{patient_name}_{dialysis_date_raw}",
            "dialysis_date": dialysis_date_raw,
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
                    except (ValueError, TypeError):
                        continue
        hr_str_raw = str(row.get("CARDIOTACH", ""))
        session["time_hr"] = [np.nan] * len(session["time_sbp"])
        if hr_str_raw and hr_str_raw != "nan":
            hr_values = hr_str_raw.split(",")
            temp_hr = []
            for h in hr_values:
                try:
                    val = float(h) if h and h != "0" else np.nan
                    temp_hr.append(val if validate_hr(val) else np.nan)
                except (ValueError, TypeError):
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
    except Exception:
        return None


def extract_idh_label(session, threshold_abs=90.0):
    """Clinical IDH definition: intradialytic SBP < 90 mmHg."""
    min_sbp = min(
        [x for x in session["time_sbp"][1:] if not np.isnan(x)], default=np.nan
    )
    if np.isnan(min_sbp):
        return "Stable"  # Default to Stable if no intradialytic measurements
    return "IDH" if min_sbp < threshold_abs else "Stable"
