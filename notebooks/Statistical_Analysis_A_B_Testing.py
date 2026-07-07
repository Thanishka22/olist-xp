# Databricks notebook source
# Statistical Analysis — A/B Testing
# Notebook: 04_statistical_analysis.py
# Purpose:  SRM check, hypothesis testing, confidence intervals,
#           p-values, CUPED variance reduction, segmentation
# Project:  OlistXP — A/B Experimentation Platform

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chisquare, norm, ttest_ind
from scipy.stats import chi2_contingency

# paths
GOLD_PATH = "dbfs:/Volumes/workspace/default/ab_esting/gold/"

# experiment config
ALPHA = 0.05   # significance level

display("Statistical analysis notebook ready")

# COMMAND ----------

# load user experiment metrics from gold
user_metrics = spark.read.format("delta").load(GOLD_PATH + "user_experiment_metrics")

display(f"Loaded: {user_metrics.count():,} users")
display(user_metrics.groupBy("variant").count())

# COMMAND ----------

# convert to pandas for statistical testing
# statistical libraries like scipy work on pandas/numpy
# not on Spark DataFrames

metrics_pd = user_metrics.toPandas()

# split into control and treatment
control   = metrics_pd[metrics_pd["variant"] == "control"]
treatment = metrics_pd[metrics_pd["variant"] == "treatment"]

print(f"Control users:   {len(control):,}")
print(f"Treatment users: {len(treatment):,}")
print(f"Total users:     {len(metrics_pd):,}")

# COMMAND ----------


# STEP 1 — Sample Ratio Mismatch (SRM) Check

# This MUST run before any statistical analysis
# If SRM is detected, all results are invalid

n_control   = len(control)
n_treatment = len(treatment)
n_total     = n_control + n_treatment

# expected counts under perfect 50/50 split
expected_control   = n_total * 0.50
expected_treatment = n_total * 0.50

# chi-square test
chi2_stat, p_value_srm = chisquare(
    f_obs=[n_control, n_treatment],
    f_exp=[expected_control, expected_treatment]
)


print("SRM CHECK")

print(f"Control users:   {n_control:,}  ({n_control/n_total*100:.1f}%)")
print(f"Treatment users: {n_treatment:,}  ({n_treatment/n_total*100:.1f}%)")
print(f"Chi2 statistic:  {chi2_stat:.4f}")
print(f"P-value:         {p_value_srm:.4f}")
print()
if p_value_srm < 0.01:
    print("SRM DETECTED — results are invalid")
    print("Investigate assignment pipeline")
else:
    print("No SRM detected — assignment looks healthy")
    print("Safe to proceed with analysis")

# COMMAND ----------


# STEP 2 — Two-proportion z-test on CVR

# H₀: CVR_treatment = CVR_control  (no effect)
# H₁: CVR_treatment ≠ CVR_control  (two-sided)

n_ctrl     = len(control)
n_trt      = len(treatment)
conv_ctrl  = control["converted"].sum()
conv_trt   = treatment["converted"].sum()

cvr_ctrl   = conv_ctrl / n_ctrl
cvr_trt    = conv_trt  / n_trt

# pooled proportion under H₀
p_pool = (conv_ctrl + conv_trt) / (n_ctrl + n_trt)

# standard error
se = np.sqrt(p_pool * (1 - p_pool) * (1/n_ctrl + 1/n_trt))

# z-statistic
z_stat = (cvr_trt - cvr_ctrl) / se

# p-value (two-sided)
p_value_cvr = 2 * (1 - norm.cdf(abs(z_stat)))

# confidence interval (95%)
se_diff  = np.sqrt((cvr_ctrl*(1-cvr_ctrl)/n_ctrl) + (cvr_trt*(1-cvr_trt)/n_trt))
ci_lower = (cvr_trt - cvr_ctrl) - 1.96 * se_diff
ci_upper = (cvr_trt - cvr_ctrl) + 1.96 * se_diff

# relative uplift
relative_uplift = (cvr_trt - cvr_ctrl) / cvr_ctrl


print("CONVERSION RATE TEST")

print(f"Control CVR:     {cvr_ctrl*100:.2f}%")
print(f"Treatment CVR:   {cvr_trt*100:.2f}%")
print(f"Absolute lift:   {(cvr_trt-cvr_ctrl)*100:+.2f}pp")
print(f"Relative lift:   {relative_uplift*100:+.2f}%")
print(f"Z-statistic:     {z_stat:.4f}")
print(f"P-value:         {p_value_cvr:.6f}")
print(f"95% CI:          [{ci_lower*100:.2f}pp, {ci_upper*100:.2f}pp]")
print()
if p_value_cvr < ALPHA:
    print(f"SIGNIFICANT — reject H₀ (p={p_value_cvr:.6f} < {ALPHA})")
    print("Treatment CVR is significantly different from control")
else:
    print(f"NOT SIGNIFICANT — fail to reject H₀ (p={p_value_cvr:.6f} > {ALPHA})")

# COMMAND ----------


# STEP 3 — Welch's t-test on Average Order Value

# H₀: AOV_treatment = AOV_control
# H₁: AOV_treatment ≠ AOV_control

# only include users who actually converted
ctrl_converted  = control[control["converted"] == 1]["total_revenue"]
trt_converted   = treatment[treatment["converted"] == 1]["total_revenue"]

t_stat, p_value_aov = ttest_ind(ctrl_converted, trt_converted, equal_var=False)

aov_ctrl = ctrl_converted.mean()
aov_trt  = trt_converted.mean()

print("=" * 50)
print("AVERAGE ORDER VALUE TEST")
print("=" * 50)
print(f"Control AOV:     R${aov_ctrl:.2f}")
print(f"Treatment AOV:   R${aov_trt:.2f}")
print(f"Difference:      R${aov_trt - aov_ctrl:+.2f}")
print(f"T-statistic:     {t_stat:.4f}")
print(f"P-value:         {p_value_aov:.6f}")
print()
if p_value_aov < ALPHA:
    print(f"SIGNIFICANT — AOV difference is real (p={p_value_aov:.6f})")
else:
    print(f"NOT SIGNIFICANT — AOV difference could be noise (p={p_value_aov:.6f})")

# COMMAND ----------

# investigate the issue
print(f"Control converted users:   {len(ctrl_converted):,}")
print(f"Treatment converted users: {len(trt_converted):,}")
print(f"Control nulls in revenue:  {ctrl_converted.isna().sum()}")
print(f"Treatment nulls in revenue:{trt_converted.isna().sum()}")
print(f"Control std:               {ctrl_converted.std()}")
print(f"Treatment std:             {trt_converted.std()}")

# COMMAND ----------

# drop nulls before t-test
ctrl_converted  = control[control["converted"] == 1]["total_revenue"].dropna()
trt_converted   = treatment[treatment["converted"] == 1]["total_revenue"].dropna()

t_stat, p_value_aov = ttest_ind(ctrl_converted, trt_converted, equal_var=False)

aov_ctrl = ctrl_converted.mean()
aov_trt  = trt_converted.mean()


print("AVERAGE ORDER VALUE TEST")

print(f"Control AOV:     R${aov_ctrl:.2f}")
print(f"Treatment AOV:   R${aov_trt:.2f}")
print(f"Difference:      R${aov_trt - aov_ctrl:+.2f}")
print(f"T-statistic:     {t_stat:.4f}")
print(f"P-value:         {p_value_aov:.6f}")
print()
if p_value_aov < ALPHA:
    print(f"SIGNIFICANT — AOV difference is real (p={p_value_aov:.6f})")
else:
    print(f"NOT SIGNIFICANT — AOV difference could be noise (p={p_value_aov:.6f})")

# COMMAND ----------


# STEP 4 — Welch's t-test on Revenue per User

# includes ALL users — converted and non-converted
# non-converted users have revenue = 0
# this is the true business impact metric

ctrl_revenue = control["total_revenue"].fillna(0)
trt_revenue  = treatment["total_revenue"].fillna(0)

t_stat_rev, p_value_rev = ttest_ind(ctrl_revenue, trt_revenue, equal_var=False)

print("REVENUE PER USER TEST")

print(f"Control rev/user:   R${ctrl_revenue.mean():.2f}")
print(f"Treatment rev/user: R${trt_revenue.mean():.2f}")
print(f"Difference:         R${trt_revenue.mean() - ctrl_revenue.mean():+.2f}")
print(f"T-statistic:        {t_stat_rev:.4f}")
print(f"P-value:            {p_value_rev:.6f}")
print()
if p_value_rev < ALPHA:
    print(f"SIGNIFICANT (p={p_value_rev:.6f})")
else:
    print(f"NOT SIGNIFICANT (p={p_value_rev:.6f})")

# COMMAND ----------


# STEP 5 — CUPED Variance Reduction

import numpy as np
from scipy.stats import ttest_ind

# CUPED uses pre-experiment data to reduce variance
# we use avg_review_score as our pre-experiment covariate
# (proxy for pre-experiment user behavior)

# drop nulls
cuped_df = metrics_pd.dropna(subset=["total_revenue", "avg_review_score"])

X = cuped_df["avg_review_score"].values   # covariate (pre-experiment)
Y = cuped_df["total_revenue"].values      # outcome metric

# calculate theta — the adjustment coefficient
theta = np.cov(Y, X)[0, 1] / np.var(X)

# apply CUPED adjustment
cuped_df = cuped_df.copy()
cuped_df["cuped_revenue"] = Y - theta * (X - np.mean(X))

# variance before and after
var_before = np.var(Y)
var_after  = np.var(cuped_df["cuped_revenue"])
reduction  = (1 - var_after / var_before) * 100

print("CUPED VARIANCE REDUCTION")

print(f"Theta:              {theta:.4f}")
print(f"Variance before:    {var_before:.4f}")
print(f"Variance after:     {var_after:.4f}")
print(f"Variance reduced:   {reduction:.1f}%")

# re-run t-test on CUPED adjusted metric
ctrl_cuped = cuped_df[cuped_df["variant"] == "control"]["cuped_revenue"]
trt_cuped  = cuped_df[cuped_df["variant"] == "treatment"]["cuped_revenue"]

t_stat_cuped, p_value_cuped = ttest_ind(ctrl_cuped, trt_cuped, equal_var=False)

print(f"\nOriginal p-value:   {p_value_rev:.6f}")
print(f"CUPED p-value:      {p_value_cuped:.6f}")
print(f"T-statistic:        {t_stat_cuped:.4f}")
print()
if p_value_cuped < ALPHA:
    print(f"SIGNIFICANT after CUPED (p={p_value_cuped:.6f})")
else:
    print(f"NOT SIGNIFICANT after CUPED (p={p_value_cuped:.6f})")

# COMMAND ----------

# STEP 6 — Segmentation Analysis
# create segments from available columns
metrics_pd["is_repeat_buyer"] = (metrics_pd["num_orders"] > 1).astype(int)
metrics_pd["has_late_delivery"] = (metrics_pd["late_delivery_rate"] > 0).astype(int)

segments = ["is_repeat_buyer", "has_late_delivery"]

print("=" * 50)
print("SEGMENTATION ANALYSIS — CVR by segment")
print("=" * 50)

for segment in segments:
    print(f"\nSegment: {segment}")
    print("-" * 40)
    
    for seg_val in sorted(metrics_pd[segment].dropna().unique()):
        seg_data = metrics_pd[metrics_pd[segment] == seg_val]
        
        ctrl_seg = seg_data[seg_data["variant"] == "control"]["converted"]
        trt_seg  = seg_data[seg_data["variant"] == "treatment"]["converted"]
        
        if len(ctrl_seg) == 0 or len(trt_seg) == 0:
            continue
            
        cvr_c = ctrl_seg.mean()
        cvr_t = trt_seg.mean()
        lift  = (cvr_t - cvr_c) * 100
        
        print(f"  {segment}={int(seg_val)}: "
              f"control={cvr_c*100:.1f}%  "
              f"treatment={cvr_t*100:.1f}%  "
              f"lift={lift:+.1f}pp  "
              f"n={len(seg_data):,}")