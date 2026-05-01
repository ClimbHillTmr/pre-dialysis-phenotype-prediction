# 血流动力学表型图鉴 (Hemodynamic Phenotype Landscape)

**Project Landscape V2** | 双中心全量数据分析报告  
**日期**: 2026-04-17  
**数据规模**: 226,800 条完整透析记录 (深医 150,959 + 福鼎 75,841)

---

## 执行摘要

本研究放弃传统的 IDH 二分类预测范式，采用**无监督时序表型发现 + 监督学习预测**的两步走战略，构建透析并发症的完整分析闭环。核心发现：

1. **四大表型**：通过 Soft-DTW K-Means 聚类识别出 4 种 ΔSBP 轨迹表型，其中 P0 (Severe Drop) 表型 IDH 发生率高达 47.0%
2. **跨中心差异**：福鼎 P0 表型高发的根本原因是 **UFR 单位语义差异**（福鼎 L/hr vs 深医 ml/hr），导致福鼎处方 UFR 是深医的 4.28 倍 (p<0.001)
3. **可预测性**：仅使用透前静态特征 + UFR 参数，LightGBM 模型预测 P0 表型 AUROC=0.623

---

## Phase 1: 权威表型提取 (Robust Phenotyping)

### 1.1 方法学

- **数据清洗**: 对双中心 226,800 条记录进行完整清洗，包括异常值检测、缺失值处理、UFR 语义标准化
- **轨迹标准化**: 将每条透析记录的 SBP 转换为 ΔSBP (相对于透前基线)
- **两阶段聚类**:
  - 阶段一：分层抽样 50,000 条，运行 Soft-DTW K-Means (K=4, 50 次迭代)
  - 阶段二：使用 FastDTW 1-NN 将剩余 176,800 条分配至 4 个聚类中心

### 1.2 表型分布

| 表型 | 名称 | 样本量 | 占比 | IDH 发生率 |
|------|------|--------|------|------------|
| **P0** | Severe Drop (末端跳水型) | 80,144 | 35.3% | **47.0%** |
| **P1** | Rise (末端升高型) | 52,274 | 23.0% | 2.5% |
| **P2** | Stable (全程平稳型) | 42,764 | 18.9% | 2.2% |
| **P3** | Severe Drop (全程跳水型) | 51,618 | 22.8% | **100.0%** |

### 1.3 表型轨迹特征

![表型轨迹图](file:///home/cht/Works/OptiHemoAI/hd_trajectory_poc/figures/Figure1_Phenotype_Landscape_Full.png)

**临床意义**:
- **P0 (Severe Drop)**: 透析中后期 SBP 急剧下降，IDH 风险极高 (47.0%)
- **P1 (Rise)**: 透析末期 SBP 反常升高，可能与代偿机制相关
- **P2 (Stable)**: 全程血压平稳，IDH 风险最低 (2.2%)
- **P3 (Severe Drop)**: 全程持续下降，IDH 发生率 100%

---

## Phase 2: 跨中心地貌剖析 (Center Landscape)

### 2.1 核心发现：福鼎 P0 泛滥真凶

| 指标 | 深医 (n=150,059) | 福鼎 (n=74,412) | 统计检验 |
|------|------------------|-----------------|----------|
| **处方 UFR (ml/hr)** | 602.3 ± 219.5 | **2,582.2 ± 837.4** | **t=-858.382, p<0.001** |
| **IDWG (kg)** | 3.03 ± 18.46 | 3.00 ± 7.46 | t=0.427, p=0.669 |

### 2.2 根因分析

**UFR 语义差异溯源**:

| 中心 | UFR 字段含义 | 单位 | 计算方式 |
|------|-------------|------|----------|
| **深医** | 实际超滤率 (时间序列) | ml/hr | 超滤量 (ml) / 实际时长 (hr) |
| **福鼎** | 处方超滤率 (单一值) | L/hr | 机器面板输入值 |

**关键发现**:
1. 福鼎 UFR 字段单位是 **L/hr**，深医是 **ml/hr**，相差 1000 倍量级
2. 福鼎处方 UFR 是深医的 **4.28 倍** (2582 vs 602 ml/hr)
3. IDWG 两中心无显著差异 (p=0.669)，排除透间体重增长因素
4. **结论**: 福鼎 P0 表型高发是由于处方超滤率设置过高，导致透析中血容量快速下降

![Root Cause Analysis](file:///home/cht/Works/OptiHemoAI/hd_trajectory_poc/figures/Figure2_Root_Cause_Analysis.png)

---

## Phase 3: 表型的前置预测 (Pre-dialysis Phenotype Prediction)

### 3.1 模型配置

- **算法**: LightGBM 多分类器 (n_estimators=300, max_depth=6)
- **输入特征**: 11 个透前可获取特征
  - 人口学: 年龄、性别
  - 体重: 透前体重、干体重、IDWG
  - 血压: 透前 SBP、透前 DBP
  - UFR 相关: 处方 UFR、实际 UFR、UFR 执行差 (处方-实际)
  - 中心标识
- **训练/测试**: 80/20 分层划分 (有效样本 1,529 条)

### 3.2 模型性能

| 指标 | 值 |
|------|-----|
| **整体准确率** | 0.458 |
| **P0 AUROC** | 0.623 |
| **P0 AUPRC** | 0.497 |
| **P1 AUROC** | 0.782 |
| **P2 AUROC** | 0.627 |
| **P3 AUROC** | 0.785 |

### 3.3 SHAP 可解释性分析 (P0 专属)

![SHAP Summary Plot](file:///home/cht/Works/OptiHemoAI/hd_trajectory_poc/figures/Figure3_SHAP_Summary_P0.png)

**P0 风险关键驱动因素**:
1. **处方 UFR**: 高处方 UFR 显著增加 P0 风险
2. **透前 SBP**: 高透前 SBP 与 P0 风险正相关
3. **UFR 执行差**: 处方与实际 UFR 差异影响 P0 风险

![UFR SHAP Special](file:///home/cht/Works/OptiHemoAI/hd_trajectory_poc/figures/Figure3_SHAP_UFR_Special.png)

![Prediction Overview](file:///home/cht/Works/OptiHemoAI/hd_trajectory_poc/figures/Figure3_Phenotype_Prediction_Full.png)

---

## 临床启示

### 4.1 对福鼎中心的建议

1. **UFR 处方审查**: 福鼎中心处方 UFR (2582 ml/hr) 显著高于深医 (602 ml/hr)，建议审查 UFR 处方规范
2. **高危筛查**: 对 UFR > 1000 ml/hr 的患者加强透析中血压监测
3. **个体化 UFR**: 根据患者透前 SBP、IDWG 等特征动态调整 UFR

### 4.2 对深医中心的建议

1. **P3 表型关注**: 深医 P3 表型 (全程跳水型) IDH 发生率 100%，需重点关注
2. **早期预警**: 利用透前特征 + LightGBM 模型提前识别 P0/P3 高危患者

### 4.3 跨中心标准化

1. **UFR 单位统一**: 建议双中心统一 UFR 记录单位为 ml/hr
2. **数据字典**: 建立统一的数据字典，明确各字段语义

---

## 方法学局限

1. **有效样本限制**: LightGBM 模型仅使用 1,529 条完整 UFR 字段样本，可能影响模型泛化能力
2. **UFR 语义差异**: 福鼎缺乏实际 UFR 时间序列，仅能用处方值近似
3. **单中心验证**: 模型未进行外部验证，泛化能力待验证
4. **表型稳定性**: 聚类结果可能受抽样随机性影响，需进行稳定性分析

---

## 输出文件清单

| 文件 | 描述 |
|------|------|
| [Figure1_Phenotype_Landscape_Full.png](file:///home/cht/Works/OptiHemoAI/hd_trajectory_poc/figures/Figure1_Phenotype_Landscape_Full.png) | 四大表型 ΔSBP 轨迹图 |
| [Figure2_Root_Cause_Analysis.png](file:///home/cht/Works/OptiHemoAI/hd_trajectory_poc/figures/Figure2_Root_Cause_Analysis.png) | UFR/IDWG 跨中心差异分析 |
| [Figure3_SHAP_Summary_P0.png](file:///home/cht/Works/OptiHemoAI/hd_trajectory_poc/figures/Figure3_SHAP_Summary_P0.png) | P0 表型 SHAP 解释性图 |
| [Figure3_SHAP_UFR_Special.png](file:///home/cht/Works/OptiHemoAI/hd_trajectory_poc/figures/Figure3_SHAP_UFR_Special.png) | UFR 特征 SHAP 专项分析 |
| [Figure3_Phenotype_Prediction_Full.png](file:///home/cht/Works/OptiHemoAI/hd_trajectory_poc/figures/Figure3_Phenotype_Prediction_Full.png) | 预测模型性能总览 |

---

## 代码与数据

- **主分析脚本**: [Project_Landscape_V2.py](file:///home/cht/Works/OptiHemoAI/hd_trajectory_poc/Project_Landscape_V2.py)
- **清洗数据缓存**: `data/processed/all_cleaned_full.pkl`
- **运行日志**: `logs/landscape_v2_final_20260417_153407.log`

---

*本报告基于双中心 226,800 条完整透析记录，采用 Soft-DTW K-Means 聚类 + LightGBM 预测 + SHAP 解释性分析，构建透析血流动力学表型完整分析闭环。*
