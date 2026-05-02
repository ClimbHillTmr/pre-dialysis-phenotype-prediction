"""
Configuration file for hemodynamic-phenotypes-hd project.
"""

# Data paths
SHENYI_DATA_DIR = "/home/cht/Works/OptiHemoAI/dataset/深医/透析数据"
FUDING_DATA_DIR = "/home/cht/Works/OptiHemoAI/dataset/2023-4-19 福鼎市医院HD数据-周鸿明整理后"
FUDING_PREPOST_CSV = "福鼎透前透后.csv"
FUDING_MID_CSV = "福鼎透中20221025-透中信息用这个表(1).csv"

# Clustering parameters
N_CLUSTERS = 4
MAX_SEQ_LEN = 51
RANDOM_STATE = 42
SAMPLE_SIZE = 50_000
KMEANS_MAX_ITER = 50
N_JOBS = -1

# Validation thresholds
MIN_TIMEPOINTS = 5
SBP_MIN = 40
SBP_MAX = 300
DBP_MIN = 20
DBP_MAX = 150
HR_MIN = 30
HR_MAX = 250
IDH_THRESHOLD_ABS = 90.0

# Visualization settings
FIGURE_DPI = 300
FIGURE_FORMAT = "pdf"
COLORS = ["#c0392b", "#27ae60", "#2980b9", "#e67e22"]
PHENOTYPE_NAMES = [
    "P0: Severe Drop",
    "P1: Moderate Drop",
    "P2: Stable",
    "P3: Mild Drop",
]
