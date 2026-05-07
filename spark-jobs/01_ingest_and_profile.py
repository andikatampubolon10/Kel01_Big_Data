from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, count, when, sum as fsum, approx_count_distinct

INPUT_CSV = "../data/raw/Online Retail Data.csv"
BRONZE_OUT = "../data/bronze/online_retail_bronze.parquet"

spark = (
    SparkSession.builder
    .appName("online-retail-ingest-profile")
    .getOrCreate()
)

# 1) Ingest CSV
df = (
    spark.read
    .option("header", "true")
    .option("multiLine", "true")     # aman kalau ada Description berisi koma/newline
    .option("escape", "\"")
    .option("quote", "\"")
    .csv(INPUT_CSV)
)

print("=== Schema (raw) ===")
df.printSchema()

print("=== Sample rows ===")
df.show(10, truncate=True)

# 2) Cast tipe data + parse timestamp (profiling butuh tipe numerik)
# Catatan: format umum dataset Online Retail biasanya "M/d/yyyy H:mm"
df_typed = (
    df
    .withColumn("Quantity", col("Quantity").cast("int"))
    .withColumn("UnitPrice", col("UnitPrice").cast("double"))
    .withColumn("CustomerID", col("CustomerID").cast("string"))
    .withColumn("InvoiceTS", to_timestamp(col("InvoiceDate"), "M/d/yyyy H:mm"))
)

# 3) Data profiling ringkas (untuk laporan & rencana cleaning)
total_rows = df_typed.count()

profile = df_typed.select(
    count("*").alias("rows"),
    approx_count_distinct("InvoiceNo").alias("distinct_invoices"),
    approx_count_distinct("CustomerID").alias("distinct_customers"),
    approx_count_distinct("StockCode").alias("distinct_products"),

    # null checks
    fsum(when(col("CustomerID").isNull(), 1).otherwise(0)).alias("null_customerid"),
    fsum(when(col("Description").isNull(), 1).otherwise(0)).alias("null_description"),
    fsum(when(col("InvoiceTS").isNull(), 1).otherwise(0)).alias("null_invoicets"),

    # anomaly checks
    fsum(when(col("Quantity") <= 0, 1).otherwise(0)).alias("qty_le_0"),
    fsum(when(col("UnitPrice") <= 0, 1).otherwise(0)).alias("unitprice_le_0"),
).collect()[0]

print("=== Profiling Summary ===")
for k, v in profile.asDict().items():
    print(f"{k}: {v}")

# 4) Simpan Bronze (raw-ish tapi sudah ada kolom InvoiceTS untuk memudahkan tahap berikutnya)
(
    df_typed
    .write
    .mode("overwrite")
    .parquet(BRONZE_OUT)
)

print(f"Bronze parquet saved to: {BRONZE_OUT}")
spark.stop()