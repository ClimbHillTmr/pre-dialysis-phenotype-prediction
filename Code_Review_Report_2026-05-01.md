# 代码审查报告：HemoDynamics & HemoPredict

**审查日期**: 2026-05-01  
**审查人**: AI Code Reviewer  
**严重级别**: 🔴 严重 | 🟠 高危 | 🟡 中危 | 🟢 低危

---

## 一、仓库一：HemoDynamics

### 1.1 安全漏洞

#### 🔴 CRITICAL-001: `eval()` 代码注入漏洞
**文件**: `src/data/loader.py:L68`  
**问题**:
```python
def convert_to_float_list(input_list):
    if isinstance(input_list, str):
        try:
            parsed_list = eval(input_list)  # ← 严重安全漏洞
```
**风险**: 恶意CSV文件可执行任意Python代码  
**修复**: 使用 `ast.literal_eval()` 替代 `eval()`

---

#### 🟠 HIGH-001: 无输入数据验证
**文件**: `src/main.py:L53-70`  
**问题**:
```python
shenyi_files = list(Path(SHENYI_DATA_DIR).glob("*.csv"))
for f in shenyi_files:
    df = pd.read_csv(f)  # ← 无文件大小、编码、格式验证
```
**风险**: 恶意或损坏的CSV文件可导致拒绝服务或数据损坏  
**修复**: 添加文件大小限制、编码验证、列名检查

---

### 1.2 架构设计缺陷

#### 🟠 HIGH-002: 硬编码输出路径
**文件**: `src/main.py:L136`  
**问题**:
```python
def run_analysis(labeled_sessions, output_dir="results"):  # ← 硬编码
```
**影响**: 无法通过配置文件修改输出路径，降低灵活性  
**修复**: 从 `config/settings.py` 读取 `OUTPUT_DIR`

---

#### 🟡 MED-001: 内存效率问题
**文件**: `src/main.py:L87-97`  
**问题**:
```python
def normalize_trajectories(all_sessions, max_len=MAX_SEQ_LEN):
    normalized = []
    for s in all_sessions:  # ← 200K+ 会话逐个处理
        pre_sbp = s["time_sbp"][0]
        delta_sbp = [x - pre_sbp for x in s["time_sbp"]]
```
**影响**: 对于大规模数据，列表推导式创建大量临时对象  
**修复**: 使用生成器表达式或numpy向量化操作

---

#### 🟡 MED-002: 无异常处理
**文件**: `src/main.py:L243-262`  
**问题**:
```python
def main():
    all_sessions = load_and_parse_data()  # ← 无try-catch
    normalized = normalize_trajectories(all_sessions)
    labeled, centroids = run_clustering(normalized)
```
**影响**: 任何阶段失败都会导致整个pipeline崩溃，无有用错误信息  
**修复**: 添加try-except块和日志记录

---

#### 🟡 MED-003: 随机种子管理混乱
**文件**: `src/main.py:L106`, `L207`  
**问题**:
```python
np.random.seed(RANDOM_STATE)  # ← 多次调用，状态不可控
sample = np.random.choice(normalized_sessions, ...)
```
**影响**: 多次设置种子可能导致不可重现的结果  
**修复**: 使用 `np.random.RandomState(RANDOM_STATE)` 创建独立RNG实例

---

### 1.3 数据处理缺陷

#### 🔴 CRITICAL-002: ΔSBP归一化基准值异常
**文件**: `src/main.py:L90`  
**问题**:
```python
pre_sbp = s["time_sbp"][0]  # ← 如果第一个值是异常值？
delta_sbp = [x - pre_sbp for x in s["time_sbp"]]
```
**影响**: 如果透前SBP是异常值（如测量错误），所有ΔSBP都会错误  
**修复**: 添加基准值验证，或使用前3个测量点的中位数

---

#### 🟠 HIGH-003: IDH定义边界情况
**文件**: `src/data/loader.py:L278-283`  
**问题**:
```python
def extract_idh_label(session, threshold_abs=90.0):
    min_sbp = min(
        [x for x in session["time_sbp"][1:] if not np.isnan(x)], default=np.nan
    )
    if np.isnan(min_sbp):
        return "unknown"  # ← 未知标签会导致下游分析错误
```
**影响**: "unknown" 标签在统计计算中会导致除零错误或NaN传播  
**修复**: 排除unknown样本或提供默认处理

---

#### 🟠 HIGH-004: 质心计算信息丢失
**文件**: `src/clustering/soft_dtw_kmeans.py:L45-52`  
**问题**:
```python
def compute_centroids(X, labels, n_clusters=N_CLUSTERS):
    for k in range(n_clusters):
        cluster_pts = [X[i] for i in range(len(X)) if labels[i] == k]
        min_len = min(len(p) for p in cluster_pts)
        truncated = [p[:min_len] for p in cluster_pts]  # ← 截断长序列
        new_centers.append(np.mean(truncated, axis=0))
```
**影响**: 截断到最小长度会丢失长序列的后期信息，导致质心不准确  
**修复**: 使用插值对齐或DTW Barycenter Averaging (DBA)

---

#### 🟡 MED-004: 表型特征化阈值无临床依据
**文件**: `src/clustering/soft_dtw_kmeans.py:L108-118`  
**问题**:
```python
def characterize_phenotype(centroid):
    final = centroid[-1]
    min_val = np.min(centroid)
    if final < -30 and min_val < -35:
        return "P0: Severe Drop"  # ← 硬编码阈值，无临床依据
```
**影响**: 阈值没有临床文献支持，可能导致错误分类  
**修复**: 基于临床文献或专家共识定义阈值，或使用数据驱动方法

---

### 1.4 统计方法缺陷

#### 🟡 MED-005: 无多重检验校正
**文件**: `src/analysis/phenotype_stats.py`  
**问题**: 多个统计检验（卡方、t检验）没有进行Bonferroni或FDR校正  
**影响**: 增加假阳性率  
**修复**: 添加 `statsmodels.stats.multitest.multipletests` 校正

---

#### 🟡 MED-006: 卡方检验假设未验证
**文件**: `src/analysis/phenotype_stats.py`  
**问题**: 没有检查期望频数是否大于5的卡方检验假设  
**影响**: 如果期望频数过小，卡方检验结果不可靠  
**修复**: 添加期望频数检查，必要时使用Fisher精确检验

---

#### 🟠 HIGH-005: 无聚类稳定性验证
**文件**: `src/clustering/soft_dtw_kmeans.py`  
**问题**: 聚类结果没有内部验证（如bootstrap稳定性检验）  
**影响**: 无法评估聚类结果的可靠性和可重复性  
**修复**: 添加bootstrap重采样稳定性分析

---

## 二、仓库二：HemoPredict

### 2.1 架构设计缺陷

#### 🔴 CRITICAL-003: 表型标签占位符导致Pipeline无法运行
**文件**: `src/main.py:L289-295`  
**问题**:
```python
def main():
    all_sessions = load_and_parse_data()
    phenotype_labels = {}  # ← 空字典，Pipeline无法运行！
    print("Note: This pipeline requires phenotype labels...")
    # results = train_center_models(all_sessions, phenotype_labels)  # ← 注释掉
```
**影响**: **Pipeline完全无法运行**，只是一个空壳  
**修复**: 实现标签加载逻辑，从Repository 1的输出文件读取

---

#### 🟠 HIGH-006: 无数据泄漏检测
**文件**: `src/data/loader.py`  
**问题**: 没有检查特征是否包含透析中或透析后的信息  
**影响**: 如果特征包含透析中BP，会导致数据泄漏和过度乐观的性能评估  
**修复**: 添加特征审计，确保只使用透前特征

---

#### 🟡 MED-007: 模型持久化缺失
**文件**: `src/model/trainer.py`  
**问题**: 训练好的模型没有保存，无法用于新数据预测  
**影响**: 每次预测都需要重新训练，无法部署  
**修复**: 使用 `joblib.dump(model, "model.pkl")` 保存模型

---

### 2.2 模型训练缺陷

#### 🔴 CRITICAL-004: KNN插补数据泄漏
**文件**: `src/model/trainer.py:L59-62`  
**问题**:
```python
def train_lightgbm(X, y, center_name="Center"):
    imputer = KNNImputer(n_neighbors=5)
    X_imputed = imputer.fit_transform(X)  # ← 在split之前fit！
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, ...)
```
**影响**: **严重数据泄漏**！测试集信息通过KNN邻居影响训练集，导致性能高估  
**修复**:
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, ...)
imputer = KNNImputer(n_neighbors=5)
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)  # 只用训练集的统计量
```

---

#### 🟠 HIGH-007: 无超参数优化
**文件**: `src/model/trainer.py:L67-75`  
**问题**:
```python
model = lgb.LGBMClassifier(
    n_estimators=N_ESTIMATORS,  # ← 固定值
    learning_rate=LEARNING_RATE,
    max_depth=MAX_DEPTH,
    num_leaves=NUM_LEAVES,
)
```
**影响**: 固定超参数可能不是最优的，导致性能下降  
**修复**: 使用Optuna或GridSearchCV进行超参数优化

---

#### 🟠 HIGH-008: 无早停机制
**文件**: `src/model/trainer.py:L76`  
**问题**:
```python
model.fit(X_train, y_train, eval_set=[(X_test, y_test)])  # ← 无early_stopping
```
**影响**: 可能过拟合，特别是对于小样本  
**修复**:
```python
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], 
          callbacks=[lgb.early_stopping(50)])
```

---

#### 🟡 MED-008: 类别不平衡处理不足
**文件**: `src/model/trainer.py:L74`  
**问题**: 仅使用 `class_weight="balanced"` 可能不足以处理严重类别不平衡  
**影响**: 少数类可能仍然预测不佳  
**修复**: 考虑SMOTE过采样或Focal Loss

---

### 2.3 特征工程缺陷

#### 🔴 CRITICAL-005: Fuding中心性别硬编码错误
**文件**: `src/data/loader.py:L127`  
**问题**:
```python
features = {
    "center": "fuding",
    "sex": 1,  # ← 硬编码为男性！所有Fuding患者都是男性？
```
**影响**: **所有Fuding患者被错误标记为男性**，导致性别特征完全失效  
**修复**: 从数据中读取真实性别信息

---

#### 🟡 MED-009: 缺失重要临床特征
**文件**: `src/data/loader.py:L74-85`  
**问题**: 特征仅包含人口统计学和基础生命体征，缺少：
- 合并症（糖尿病、心血管疾病等）
- 血管通路类型
- 透析龄
- 用药情况
**影响**: 模型预测能力受限  
**修复**: 添加可用临床特征

---

#### 🟡 MED-010: 特征缩放缺失
**文件**: `src/data/loader.py`  
**问题**: 没有对连续特征进行标准化  
**影响**: 对于基于距离的模型（如KNN插补），量纲差异会影响结果  
**修复**: 添加StandardScaler或RobustScaler

---

### 2.4 评估缺陷

#### 🟠 HIGH-009: 无外部验证
**文件**: `src/main.py`  
**问题**: 只有内部train/test split，没有真正的外部验证集  
**影响**: 无法评估模型在未知数据上的泛化能力  
**修复**: 保留一个中心作为完全独立的外部验证集

---

#### 🟡 MED-011: 校准曲线过于简单
**文件**: `src/model/trainer.py:L176-191`  
**问题**:
```python
prob_true, prob_pred = calibration_curve(y_binary, y_prob[:, i], n_bins=10)
```
**影响**: 对于大样本（100K+），10个bins可能不够精细  
**修复**: 使用 `n_bins=20` 或自适应bins

---

#### 🟡 MED-012: 无不确定性量化
**文件**: `src/model/trainer.py`  
**问题**: 没有置信区间或bootstrap验证  
**影响**: 无法评估性能指标的统计显著性  
**修复**: 添加bootstrap重采样计算95% CI

---

### 2.5 跨中心验证缺陷

#### 🟠 HIGH-010: 特征分布差异未处理
**文件**: `src/experiments/ablation.py`  
**问题**: 两个中心的特征可能有不同的分布和缺失模式，直接应用模型可能导致错误  
**影响**: 跨中心性能下降可能被低估  
**修复**: 添加特征分布检查和领域适应技术

---

## 三、修复优先级

| 优先级 | 编号 | 问题 | 严重级别 |
|--------|------|------|----------|
| P0 | CRITICAL-001 | `eval()` 代码注入漏洞 | 🔴 严重 |
| P0 | CRITICAL-004 | KNN插补数据泄漏 | 🔴 严重 |
| P0 | CRITICAL-005 | Fuding性别硬编码错误 | 🔴 严重 |
| P0 | CRITICAL-003 | Pipeline无法运行 | 🔴 严重 |
| P1 | CRITICAL-002 | ΔSBP归一化基准值异常 | 🔴 严重 |
| P1 | HIGH-003 | IDH定义边界情况 | 🟠 高危 |
| P1 | HIGH-004 | 质心计算信息丢失 | 🟠 高危 |
| P1 | HIGH-005 | 无聚类稳定性验证 | 🟠 高危 |
| P1 | HIGH-006 | 无数据泄漏检测 | 🟠 高危 |
| P1 | HIGH-007 | 无超参数优化 | 🟠 高危 |
| P1 | HIGH-008 | 无早停机制 | 🟠 高危 |
| P1 | HIGH-009 | 无外部验证 | 🟠 高危 |
| P1 | HIGH-010 | 特征分布差异未处理 | 🟠 高危 |
| P2 | HIGH-001 | 无输入数据验证 | 🟠 高危 |
| P2 | HIGH-002 | 硬编码输出路径 | 🟠 高危 |
| P2 | MED-001 | 内存效率问题 | 🟡 中危 |
| P2 | MED-002 | 无异常处理 | 🟡 中危 |
| P2 | MED-003 | 随机种子管理混乱 | 🟡 中危 |
| P2 | MED-004 | 表型特征化阈值无临床依据 | 🟡 中危 |
| P2 | MED-005 | 无多重检验校正 | 🟡 中危 |
| P2 | MED-006 | 卡方检验假设未验证 | 🟡 中危 |
| P2 | MED-007 | 模型持久化缺失 | 🟡 中危 |
| P2 | MED-008 | 类别不平衡处理不足 | 🟡 中危 |
| P2 | MED-009 | 缺失重要临床特征 | 🟡 中危 |
| P2 | MED-010 | 特征缩放缺失 | 🟡 中危 |
| P2 | MED-011 | 校准曲线过于简单 | 🟡 中危 |
| P2 | MED-012 | 无不确定性量化 | 🟡 中危 |

---

## 四、总结

### 4.1 关键发现

**仓库一（HemoDynamics）**:
- 1个严重安全漏洞（`eval()` 注入）
- 1个严重数据处理缺陷（ΔSBP归一化）
- 5个高危设计缺陷
- 6个中危问题

**仓库二（HemoPredict）**:
- 2个严重设计缺陷（Pipeline无法运行、性别硬编码错误）
- 1个严重数据泄漏问题（KNN插补）
- 6个高危问题
- 6个中危问题

### 4.2 建议

1. **立即修复** P0级别问题（4个），这些问题会导致：
   - 安全漏洞
   - 数据泄漏
   - 错误结果
   - Pipeline无法运行

2. **优先修复** P1级别问题（8个），这些问题会影响：
   - 结果可靠性
   - 模型性能
   - 临床有效性

3. **计划修复** P2级别问题（12个），这些问题会影响：
   - 代码质量
   - 可维护性
   - 统计严谨性

### 4.3 总体评价

两个仓库在**学术投稿前**需要重大修复。特别是：
- **数据泄漏问题**（CRITICAL-004）会导致所有性能指标无效
- **性别硬编码错误**（CRITICAL-005）会导致特征工程完全失效
- **Pipeline无法运行**（CRITICAL-003）使仓库二成为空壳

建议在修复所有P0和P1问题后，重新运行所有实验并更新技术报告。

---

**报告生成时间**: 2026-05-01  
**审查状态**: 完成
