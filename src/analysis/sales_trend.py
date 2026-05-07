from pyspark.sql import DataFrame
from pyspark.sql.functions import col, to_date, date_format, sum as fsum, countDistinct

def sales_daily(df_sales: DataFrame) -> DataFrame:
    """
    Agregasi tren penjualan harian per country.
    """
    return (
        df_sales
        .withColumn("SaleDate", to_date(col("InvoiceTS")))
        .groupBy("SaleDate", "Country")
        .agg(
            fsum(col("TotalAmount")).alias("Revenue"),
            countDistinct(col("InvoiceNo")).alias("Transactions"),
            fsum(col("Quantity")).alias("ItemsSold"),
        )
        .orderBy(col("SaleDate").asc())
    )

def sales_monthly(df_sales: DataFrame) -> DataFrame:
    """
    Agregasi tren bulanan (YYYY-MM) per country.
    """
    return (
        df_sales
        .withColumn("YearMonth", date_format(col("InvoiceTS"), "yyyy-MM"))
        .groupBy("YearMonth", "Country")
        .agg(
            fsum(col("TotalAmount")).alias("Revenue"),
            countDistinct(col("InvoiceNo")).alias("Transactions"),
            fsum(col("Quantity")).alias("ItemsSold"),
        )
        .orderBy(col("YearMonth").asc())
    )