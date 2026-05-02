"""
Configuration file for pre-dialysis-phenotype-prediction project.
"""

# Data paths
SHENYI_DATA_DIR = "/home/cht/Works/OptiHemoAI/dataset/深医/透析数据"
FUDING_DATA_DIR = (
    "/home/cht/Works/OptiHemoAI/dataset/2023-4-19 福鼎市医院HD数据-周鸿明整理后"
)
FUDING_PREPOST_CSV = "福鼎透前透后.csv"
FUDING_MID_CSV = "福鼎透中20221025-透中信息用这个表(1).csv"

# Model parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 200
LEARNING_RATE = 0.05
MAX_DEPTH = 6
NUM_LEAVES = 31

# Feature selection
FEATURE_NAMES_BASE = [
    "age",
    "sex",
    "pre_sbp",
    "pre_dbp",
    "prescribed_ufr",
    "idwg",
    "dialysis_duration",
    "dry_weight",
    "pre_weight",
]

# Visualization settings
FIGURE_DPI = 300
FIGURE_FORMAT = "pdf"
