from pyspark.sql import DataFrame
from pyspark.sql.functions import col
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, DoubleType
)

def read_online_retail_csv(spark, csv_path: str) -> DataFrame:
    """
    Baca CSV mentah Online Retail.
    Schema dibuat eksplisit agar stabil.
    """
    schema = StructType([
        StructField("InvoiceNo", StringType(), True),
        StructField("StockCode", StringType(), True),
        StructField("Description", StringType(), True),
        StructField("Quantity", IntegerType(), True),
        StructField("InvoiceDate", StringType(), True),
        StructField("UnitPrice", DoubleType(), True),
        StructField("CustomerID", StringType(), True),
        StructField("Country", StringType(), True),
    ])

    df = (
        spark.read
        .option("header", "true")
        .option("multiLine", "true")
        .option("quote", "\"")
        .option("escape", "\"")
        .schema(schema)
        .csv(csv_path)
    )

    # Trim spasi (kadang ada spasi di Description/Country)
    df = (
        df.withColumn("InvoiceNo", col("InvoiceNo"))
          .withColumn("StockCode", col("StockCode"))
          .withColumn("CustomerID", col("CustomerID"))
          .withColumn("Country", col("Country"))
    )
    return df