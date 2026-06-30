# Databricks notebook source
# Gold Layer — Experiment Assignment & Metric Building
# Purpose:  Assign users to control/treatment groups
#           Simulate treatment effects
#           Build experiment metrics table
# Project:  OlistXP — A/B Experimentation Platform

from pyspark.sql.functions import (
    col, when, lit, rand, udf,
    count, sum as spark_sum,
    avg, max, min, countDistinct
)
from pyspark.sql.types import StringType
import hashlib

# paths
SILVER_PATH = "dbfs:/Volumes/workspace/default/ab_esting/silver/"
GOLD_PATH   = "dbfs:/Volumes/workspace/default/ab_esting/gold/"

# experiment config
EXPERIMENT_ID   = "checkout_simplification_v1"
EXPERIMENT_SALT = "olist_exp_2024"
TREATMENT_PCT   = 50
EXP_START_DATE  = "2018-01-01"
EXP_END_DATE    = "2018-01-31"

# create gold directory
dbutils.fs.mkdirs("/Volumes/workspace/default/ab_esting/gold/")

display("Gold notebook ready")

# COMMAND ----------

# load silver master table
order_fact = spark.read.format("delta").load(SILVER_PATH + "order_fact_featured")

display(f"Loaded: {order_fact.count():,} rows")
display(f"Columns: {len(order_fact.columns)}")

# COMMAND ----------

# filter to experiment window
# we only assign users who ordered during this period

experiment_orders = order_fact.filter(
    (col("order_purchase_timestamp") >= EXP_START_DATE) &
    (col("order_purchase_timestamp") <= EXP_END_DATE)
)

display(f"Total orders:      {order_fact.count():,}")
display(f"Experiment orders: {experiment_orders.count():,}")

# COMMAND ----------

# get unique users in experiment window
eligible_users = (
    experiment_orders
    .select("customer_unique_id")
    .distinct()
)

display(f"Eligible users: {eligible_users.count():,}")

# COMMAND ----------

# experiment assignment function
def assign_variant(customer_unique_id):
    if not customer_unique_id:
        return "control"
    
    hash_input = f"{customer_unique_id}_{EXPERIMENT_ID}_{EXPERIMENT_SALT}"
    hex_digest  = hashlib.md5(hash_input.encode("utf-8")).hexdigest()
    hash_int    = int(hex_digest, 16)
    bucket      = hash_int % 100
    
    return "treatment" if bucket < TREATMENT_PCT else "control"

# register as spark UDF
assign_variant_udf = udf(assign_variant, StringType())

# apply to eligible users
assignments = eligible_users.withColumn(
    "variant",
    assign_variant_udf(col("customer_unique_id"))
)

# check split
assignments.groupBy("variant").count().show()

# COMMAND ----------

(
    assignments.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .save(GOLD_PATH + "experiment_assignments")
)

display(f"Assignments saved: {assignments.count():,} users")

# COMMAND ----------

# join assignments back to experiment orders
experiment_data = experiment_orders.join(
    assignments,
    on="customer_unique_id",
    how="inner"
)

display(f"Experiment data rows: {experiment_data.count():,}")
display(experiment_data.groupBy("variant").count())

# COMMAND ----------

# Step 6 — Simulate treatment effects

BASELINE_CVR        = 0.75    # realistic baseline conversion rate
TREATMENT_CVR_LIFT  = 0.04    # treatment converts 4% more (absolute)
TREATMENT_AOV_CHANGE = -0.02  # treatment AOV drops 2% (less upsell)

# add a random number per row — deterministic with seed
experiment_data_sim = experiment_data.withColumn("rand_val", rand(seed=42))

# simulate conversion based on variant
experiment_data_sim = experiment_data_sim.withColumn(
    "sim_converted",
    when(
        col("variant") == "treatment",
        when(col("rand_val") < BASELINE_CVR + TREATMENT_CVR_LIFT, 1).otherwise(0)
    ).otherwise(
        when(col("rand_val") < BASELINE_CVR, 1).otherwise(0)
    )
)

# simulate revenue based on variant and conversion
experiment_data_sim = experiment_data_sim.withColumn(
    "sim_revenue",
    when(
        col("sim_converted") == 1,
        when(
            col("variant") == "treatment",
            col("total_order_value") * (1 + TREATMENT_AOV_CHANGE)
        ).otherwise(col("total_order_value"))
    ).otherwise(0)
)

display(
    experiment_data_sim.groupBy("variant").agg(
        count("*").alias("n_orders"),
        spark_sum("sim_converted").alias("n_converted"),
        avg("sim_converted").alias("cvr"),
        avg("sim_revenue").alias("avg_revenue")
    )
)

# COMMAND ----------

# Step 7 — Build per-user experiment metrics table

user_metrics = experiment_data_sim.groupBy(
    "customer_unique_id", "variant"
).agg(
    max("sim_converted").alias("converted"),
    spark_sum("sim_revenue").alias("total_revenue"),
    count("order_id").alias("num_orders"),
    avg("review_score").alias("avg_review_score"),
    avg("days_to_delivery").alias("avg_delivery_days"),
    avg("is_late").alias("late_delivery_rate")
)

display(f"User metrics rows: {user_metrics.count():,}")
display(
    user_metrics.groupBy("variant").agg(
        count("*").alias("n_users"),
        spark_sum("converted").alias("n_converted"),
        avg("converted").alias("cvr"),
        avg("total_revenue").alias("avg_revenue_per_user"),
        avg("avg_review_score").alias("avg_satisfaction")
    )
)

# COMMAND ----------

(
    user_metrics.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .save(GOLD_PATH + "user_experiment_metrics")
)

display(f"Saved: {user_metrics.count():,} users")