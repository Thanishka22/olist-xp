# Databricks notebook source
spark.range(10).show()

# COMMAND ----------

display(dbutils.fs.ls("/Volumes/workspace/default/ab_esting"))

# COMMAND ----------

Base = "dbfs:/Volumes/workspace/default/ab_esting/archive-2/" 
orders = spark.read.csv(Base + "olist_orders_dataset.csv", header=True, inferSchema=True)
customers = spark.read.csv(Base + "olist_customers_dataset.csv", header=True, inferSchema=True)
order_items = spark.read.csv(Base + "olist_order_items_dataset.csv", header=True, inferSchema=True)
payments = spark.read.csv(Base + "olist_order_payments_dataset.csv", header=True, inferSchema=True)
reviews = spark.read.csv(Base + "olist_order_reviews_dataset.csv", header=True, inferSchema=True)
products = spark.read.csv(Base + "olist_products_dataset.csv", header=True, inferSchema=True)
category = spark.read.csv(Base + "product_category_name_translation.csv", header=True, inferSchema=True)
sellers = spark.read.csv(Base + "olist_sellers_dataset.csv", header=True, inferSchema=True)
geo = spark.read.csv(Base + "olist_geolocation_dataset.csv", header=True, inferSchema=True)

# COMMAND ----------

def explore(df, name):
    print(f" Table: {name}")
    print(f" Rows: {df.count():,}")
    print(f" Columns: {len(df.columns)}")
    print(f"schema:")
    df.printSchema()
    print(f" \n Sample Rows:")
    df.show(5, truncate= False)
    print(f"\n Null counts:")
    from pyspark.sql.functions import col, sum as spark_sum, isnan, when
    df.select([
        spark_sum(when(col(c).isNull(),1).otherwise(0)).alias(c) for c in df.columns]).show()
for df, name in [
    (orders, "orders"),
    (customers, "customers"),
    (order_items, "order_items"),
    (payments, "payments"),
    (reviews, "reviews"),
    (products, "products"),
    (category, "category_translation"),
    (sellers, "sellers"),
    (geo, "geolocation"),
]:
    explore(df, name)

# COMMAND ----------

# ── Bronze Layer: Raw Ingestion ──────────────────────────────
# Notebook: 02_bronze_ingestion.py
# Purpose:  Read raw CSVs → enforce schema → write to Delta Lake
# Author:   Your Name
# Project:  OlistXP — A/B Experimentation Platform

# COMMAND ----------

from pyspark.sql.functions import (current_timestamp, input_file_name, lit, col, when)
from pyspark.sql.types import (StructType, StructField, StringType, IntegerType, DoubleType,TimestampType)
BASE = "dbfs:/Volumes/workspace/default/ab_esting/archive-2/"
BRONZE_PATH = "dbfs:/Volumes/workspace/default/ab_esting/bronze/"
print("Imports loaded successfully")
print(f"Source path:  {BASE}")
print(f"Bronze path:  {BRONZE_PATH}")

# COMMAND ----------

# ── Explicit schemas ─────────────────────────────────────────
# Why: inferSchema scans the file twice and sometimes gets
# types wrong. Explicit schemas are faster and more reliable.

# COMMAND ----------

schema_orders = StructType([
    StructField("order_id",                      StringType(),    False),
    StructField("customer_id",                   StringType(),    False),
    StructField("order_status",                  StringType(),    True),
    StructField("order_purchase_timestamp",      TimestampType(), True),
    StructField("order_approved_at",             TimestampType(), True),
    StructField("order_delivered_carrier_date",  TimestampType(), True),
    StructField("order_delivered_customer_date", TimestampType(), True),
    StructField("order_estimated_delivery_date", TimestampType(), True),
])
schema_customers = StructType([
    StructField("customer_id",               StringType(),  False),
    StructField("customer_unique_id",        StringType(),  False),
    StructField("customer_zip_code_prefix",  IntegerType(), True),
    StructField("customer_city",             StringType(),  True),
    StructField("customer_state",            StringType(),  True),
])
schema_order_items = StructType([
    StructField("order_id",             StringType(),    False),
    StructField("order_item_id",        IntegerType(),   False),
    StructField("product_id",           StringType(),    True),
    StructField("seller_id",            StringType(),    True),
    StructField("shipping_limit_date",  TimestampType(), True),
    StructField("price",                DoubleType(),    True),
    StructField("freight_value",        DoubleType(),    True),
])

schema_payments = StructType([
    StructField("order_id",              StringType(),  False),
    StructField("payment_sequential",    IntegerType(), True),
    StructField("payment_type",          StringType(),  True),
    StructField("payment_installments",  IntegerType(), True),
    StructField("payment_value",         DoubleType(),  True),
])

schema_reviews = StructType([
    StructField("review_id",               StringType(), True),
    StructField("order_id",                StringType(), True),
    StructField("review_score",            StringType(), True),  
    # kept as string intentionally — we cast it in Silver
    StructField("review_comment_title",    StringType(), True),
    StructField("review_comment_message",  StringType(), True),
    StructField("review_creation_date",    StringType(), True),
    StructField("review_answer_timestamp", StringType(), True),
])

schema_products = StructType([
    StructField("product_id",                  StringType(),  False),
    StructField("product_category_name",        StringType(),  True),
    StructField("product_name_lenght",          IntegerType(), True),
    StructField("product_description_lenght",   IntegerType(), True),
    StructField("product_photos_qty",           IntegerType(), True),
    StructField("product_weight_g",             IntegerType(), True),
    StructField("product_length_cm",            IntegerType(), True),
    StructField("product_height_cm",            IntegerType(), True),
    StructField("product_width_cm",             IntegerType(), True),
])
# Note: keeping misspelled column names exactly as they are
# in the source. We rename them cleanly in Silver layer.

schema_sellers = StructType([
    StructField("seller_id",               StringType(),  False),
    StructField("seller_zip_code_prefix",  IntegerType(), True),
    StructField("seller_city",             StringType(),  True),
    StructField("seller_state",            StringType(),  True),
])

schema_geo = StructType([
    StructField("geolocation_zip_code_prefix",  IntegerType(), True),
    StructField("geolocation_lat",              DoubleType(),  True),
    StructField("geolocation_lng",              DoubleType(),  True),
    StructField("geolocation_city",             StringType(),  True),
    StructField("geolocation_state",            StringType(),  True),
])

schema_category = StructType([
    StructField("product_category_name",         StringType(), True),
    StructField("product_category_name_english",  StringType(), True),
])

print("All schemas defined successfully")

# COMMAND ----------

def ingest_to_bronze(file_name, schema, table_name,
                     primary_key_col):

    print(f"\n── Ingesting {table_name} ──────────────────")

    # Step 1: Read CSV
    df = (
        spark.read
        .option("header", "true")
        .option("mode", "PERMISSIVE")
        .schema(schema)
        .csv(BASE + file_name)
    )

    raw_count = df.count()
    print(f"   Rows read from CSV:   {raw_count:,}")

    # Step 2: Add audit columns
    df = (
        df
        .withColumn("_ingestion_timestamp", current_timestamp())
        .withColumn("_source_file", lit(BASE + file_name))
        .withColumn("_bronze_layer", lit("bronze"))
    )

    # Step 3: Flag corrupt rows
    df = df.withColumn(
        "_is_corrupt",
        when(col(primary_key_col).isNull(), True).otherwise(False)
    )

    corrupt_count = df.filter(col("_is_corrupt") == True).count()
    print(f"   Corrupt rows flagged: {corrupt_count:,}")
    print(f"   Clean rows:           {raw_count - corrupt_count:,}")

    # Step 4: Write to Delta Lake
    (
        df.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .save(BRONZE_PATH + table_name)
    )

    print(f"   Saved to: {BRONZE_PATH + table_name}")
    print(f"   Status:   DONE")

    return df

# COMMAND ----------

BASE        = "dbfs:/Volumes/workspace/default/ab_esting/archive-2/"
BRONZE_PATH = "dbfs:/Volumes/workspace/default/ab_esting/bronze/"

# read orders
df_test = (
    spark.read
    .option("header", "true")
    .option("mode", "PERMISSIVE")
    .csv(BASE + "olist_orders_dataset.csv")
)

print(f"Rows read: {df_test.count():,}")

# save to bronze
(
    df_test.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .save(BRONZE_PATH + "orders")
)

print("Saved successfully")

# read back to confirm
df_check = spark.read.format("delta").load(BRONZE_PATH + "orders")
print(f"Rows in Delta: {df_check.count():,}")

# COMMAND ----------

# confirm orders bronze table exists and has correct rows
df_check = spark.read.format("delta").load(BRONZE_PATH + "orders")
print(f"Row count: {df_check.count():,}")
df_check.show(3)

# COMMAND ----------

tables_to_ingest = [
    ("olist_customers_dataset.csv",           "customers"),
    ("olist_order_items_dataset.csv",         "order_items"),
    ("olist_order_payments_dataset.csv",      "payments"),
    ("olist_order_reviews_dataset.csv",       "reviews"),
    ("olist_products_dataset.csv",            "products"),
    ("olist_sellers_dataset.csv",             "sellers"),
    ("olist_geolocation_dataset.csv",         "geolocation"),
    ("product_category_name_translation.csv", "category_translation"),
]

for file_name, table_name in tables_to_ingest:
    df = (
        spark.read
        .option("header", "true")
        .option("mode", "PERMISSIVE")
        .csv(BASE + file_name)
    )
    (
        df.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .save(BRONZE_PATH + table_name)
    )
    display(f"Done — {table_name}")

# COMMAND ----------

tables = [
    ("orders",                99_441),
    ("customers",             99_441),
    ("order_items",          112_650),
    ("payments",             103_886),
    ("reviews",              104_162),
    ("products",              32_951),
    ("sellers",                3_095),
    ("geolocation",        1_000_163),
    ("category_translation",      71),
]

for table_name, expected in tables:
    df = spark.read.format("delta").load(BRONZE_PATH + table_name)
    actual = df.count()
    status = "PASS" if actual == expected else "FAIL"
    display(f"{status} — {table_name} — {actual:,} rows")

# COMMAND ----------

# Silver Layer — Step 1: Clean Reviews
# read from bronze, not from CSV
# bronze is our source of truth now

BRONZE_PATH = "dbfs:/Volumes/workspace/default/ab_esting/bronze/"
SILVER_PATH = "dbfs:/Volumes/workspace/default/ab_esting/silver/"

# create silver directory
dbutils.fs.mkdirs("/Volumes/workspace/default/ab_esting/silver/")

from pyspark.sql.functions import col, count, when
from pyspark.sql.types import IntegerType

reviews_raw = spark.read.format("delta").load(BRONZE_PATH + "reviews")

display(f"Raw reviews rows: {reviews_raw.count():,}")

# COMMAND ----------

# drop rows where order_id is null
reviews_clean = reviews_raw.filter(col("order_id").isNotNull())

dropped = reviews_raw.count() - reviews_clean.count()
display(f"Rows dropped:   {dropped:,}")
display(f"Rows remaining: {reviews_clean.count():,}")

# COMMAND ----------

# look at what is actually in review_score column
reviews_clean.groupBy("review_score").count().orderBy("count", ascending=False).show(20)

# COMMAND ----------

from pyspark.sql.functions import expr, col
from pyspark.sql.types import IntegerType

# start fresh from raw bronze data
reviews_raw = spark.read.format("delta").load(BRONZE_PATH + "reviews")

# step 1 — drop null order_id rows
reviews_clean = reviews_raw.filter(col("order_id").isNotNull())

# step 2 — use try_cast directly, NOT cast
# try_cast turns anything that is not 1-5 into null
reviews_clean = reviews_clean.withColumn(
    "review_score",
    expr("try_cast(review_score as int)")
)

# step 3 — keep only useful columns
reviews_clean = reviews_clean.select(
    "order_id",
    "review_score",
    "review_creation_date",
    "review_answer_timestamp"
)

# check result
reviews_clean.groupBy("review_score") \
             .count() \
             .orderBy("review_score") \
             .show()

# COMMAND ----------

# check review_score type
display(f"review_score type: {dict(reviews_clean.dtypes)['review_score']}")

# check score distribution
display(reviews_clean.groupBy("review_score").count().orderBy("review_score"))

# COMMAND ----------

# fix the 0 value — replace with null
reviews_clean = reviews_clean.withColumn(
    "review_score",
    when(col("review_score") == 0, None).otherwise(col("review_score"))
)

display(reviews_clean.groupBy("review_score").count().orderBy("review_score"))

# COMMAND ----------

(
    reviews_clean.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .save(SILVER_PATH + "reviews_clean")
)

df_check = spark.read.format("delta").load(SILVER_PATH + "reviews_clean")
display(f"Reviews saved: {df_check.count():,} rows")

# COMMAND ----------

display(spark.read.format("delta").load(SILVER_PATH + "reviews_clean"))

# COMMAND ----------

# Step 2 — load order_items from bronze
order_items_raw = spark.read.format("delta").load(BRONZE_PATH + "order_items")
display(order_items_raw)

# COMMAND ----------

# aggregate order_items → one row per order
from pyspark.sql.functions import col, when, expr, count, countDistinct, sum as spark_sum, avg
order_items_agg = order_items_raw.groupBy("order_id").agg(
    spark_sum("price").alias("total_item_price"),
    spark_sum("freight_value").alias("total_freight"),
    count("order_item_id").alias("num_items"),
    countDistinct("product_id").alias("num_unique_products"),
    countDistinct("seller_id").alias("num_sellers")
)

display(f"Before: {order_items_raw.count():,} rows")
display(f"After:  {order_items_agg.count():,} rows")
display(order_items_agg)

# COMMAND ----------

(
    order_items_agg.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .save(SILVER_PATH + "order_items_clean")
)

display(f"order_items saved: {order_items_agg.count():,} rows")

# COMMAND ----------

# load payments from bronze
payments_raw = spark.read.format("delta").load(BRONZE_PATH + "payments")

display(f"Before: {payments_raw.count():,} rows")

# COMMAND ----------

# aggregate payments → one row per order
payments_agg = payments_raw.groupBy("order_id").agg(
    spark_sum("payment_value").alias("total_payment_value"),
    count("payment_sequential").alias("num_payments"),
    countDistinct("payment_type").alias("num_payment_types"),
    avg("payment_installments").alias("avg_installments")
)

display(f"Before: {payments_raw.count():,} rows")
display(f"After:  {payments_agg.count():,} rows")
display(payments_agg)

# COMMAND ----------

(
    payments_agg.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .save(SILVER_PATH + "payments_clean")
)

display(f"payments saved: {payments_agg.count():,} rows")

# COMMAND ----------

# load products from bronze
products_raw = spark.read.format("delta").load(BRONZE_PATH + "products")

display(f"Products rows: {products_raw.count():,}")
display(products_raw)

# COMMAND ----------

# fix misspelled column names
products_clean = products_raw \
    .withColumnRenamed("product_name_lenght", "product_name_length") \
    .withColumnRenamed("product_description_lenght", "product_description_length")

# confirm columns are renamed
display(products_clean.columns)

# COMMAND ----------

# load category translation from bronze
category_raw = spark.read.format("delta").load(BRONZE_PATH + "category_translation")

# join English category names onto products
products_clean = products_clean.join(
    category_raw,
    on="product_category_name",
    how="left"
)

display(f"Products rows: {products_clean.count():,}")
display(products_clean)

# COMMAND ----------

(
    products_clean.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .save(SILVER_PATH + "products_clean")
)

display(f"products saved: {products_clean.count():,} rows")

# COMMAND ----------

# load all tables
orders_df      = spark.read.format("delta").load(BRONZE_PATH + "orders")
customers_df   = spark.read.format("delta").load(BRONZE_PATH + "customers")
order_items_df = spark.read.format("delta").load(SILVER_PATH + "order_items_clean")
payments_df    = spark.read.format("delta").load(SILVER_PATH + "payments_clean")
reviews_df     = spark.read.format("delta").load(SILVER_PATH + "reviews_clean")
products_df    = spark.read.format("delta").load(SILVER_PATH + "products_clean")
sellers_df     = spark.read.format("delta").load(BRONZE_PATH + "sellers")

display("All tables loaded")

# COMMAND ----------

# Step 5 — merge all tables into one master table
order_fact = (
    orders_df

    # join customers → brings in customer_unique_id, city, state
    .join(customers_df, on="customer_id", how="left")

    # join order_items → brings in total price, freight, num items
    .join(order_items_df, on="order_id", how="left")

    # join payments → brings in total payment value
    .join(payments_df, on="order_id", how="left")

    # join reviews → brings in review score
    .join(reviews_df, on="order_id", how="left")
)

display(f"order_fact rows:    {order_fact.count():,}")
display(f"order_fact columns: {len(order_fact.columns)}")
display(order_fact)

# COMMAND ----------

# check for duplicate order_ids in each table
display(f"orders duplicates: {orders_df.count() - orders_df.select('order_id').distinct().count()}")
display(f"customers duplicates: {customers_df.count() - customers_df.select('customer_id').distinct().count()}")
display(f"order_items duplicates: {order_items_df.count() - order_items_df.select('order_id').distinct().count()}")
display(f"payments duplicates: {payments_df.count() - payments_df.select('order_id').distinct().count()}")
display(f"reviews duplicates: {reviews_df.count() - reviews_df.select('order_id').distinct().count()}")

# COMMAND ----------

from pyspark.sql.window import Window
from pyspark.sql.functions import row_number

# keep only the most recent review per order
window = Window.partitionBy("order_id").orderBy(col("review_answer_timestamp").desc())

reviews_deduped = (
    reviews_df
    .withColumn("row_num", row_number().over(window))
    .filter(col("row_num") == 1)
    .drop("row_num")
)

display(f"Reviews before dedup: {reviews_df.count():,}")
display(f"Reviews after dedup:  {reviews_deduped.count():,}")

# COMMAND ----------

# Step 5 — merge all tables using deduplicated reviews
order_fact = (
    orders_df

    # join customers
    .join(customers_df, on="customer_id", how="left")

    # join order_items
    .join(order_items_df, on="order_id", how="left")

    # join payments
    .join(payments_df, on="order_id", how="left")

    # join deduplicated reviews
    .join(reviews_deduped, on="order_id", how="left")
)

display(f"order_fact rows:    {order_fact.count():,}")
display(f"order_fact columns: {len(order_fact.columns)}")

# COMMAND ----------

(
    order_fact.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .save(SILVER_PATH + "order_fact")
)

display(f"order_fact saved: {order_fact.count():,} rows")

# COMMAND ----------

# Step 6 — Feature Engineering
from pyspark.sql.functions import (
    datediff, when, col, hour,
    dayofweek, rank, lit
)
from pyspark.sql.window import Window

# load order_fact from silver
order_fact = spark.read.format("delta").load(SILVER_PATH + "order_fact")

display(f"Loaded: {order_fact.count():,} rows")

# COMMAND ----------

# Step 6 — add all feature columns

# window for ranking orders per customer by date
# needed for is_first_order
customer_window = Window.partitionBy("customer_unique_id") \
                        .orderBy("order_purchase_timestamp")

order_fact_featured = (
    order_fact

    # is_converted → did order get delivered? 1 or 0
    # this is our PRIMARY experiment metric
    .withColumn("is_converted",
        when(col("order_status") == "delivered", 1).otherwise(0))

    # days_to_delivery → days from purchase to delivery
    .withColumn("days_to_delivery",
        datediff(
            col("order_delivered_customer_date"),
            col("order_purchase_timestamp")
        ))

    # is_late → was delivery after estimated date?
    .withColumn("is_late",
        when(
            col("order_delivered_customer_date") >
            col("order_estimated_delivery_date"), 1
        ).otherwise(0))

    # total_order_value → item price + freight
    .withColumn("total_order_value",
        col("total_item_price") + col("total_freight"))

    # freight_ratio → what % of order value is freight
    .withColumn("freight_ratio",
        when(col("total_item_price") > 0,
             col("total_freight") / col("total_item_price")
        ).otherwise(None))

    # is_weekend → was order placed on weekend?
    # 1 = Sunday, 7 = Saturday in Spark
    .withColumn("is_weekend",
        when(dayofweek(col("order_purchase_timestamp"))
             .isin([1, 7]), 1).otherwise(0))

    # order_hour → what hour of day was order placed
    .withColumn("order_hour",
        hour(col("order_purchase_timestamp")))

    # is_first_order → is this customer's first ever order?
    .withColumn("is_first_order",
        when(rank().over(customer_window) == 1, 1).otherwise(0))
)

display(f"Rows:    {order_fact_featured.count():,}")
display(f"Columns: {len(order_fact_featured.columns)}")
display(order_fact_featured)

# COMMAND ----------

(
    order_fact_featured.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .save(SILVER_PATH + "order_fact_featured")
)

display(f"order_fact_featured saved: {order_fact_featured.count():,} rows")