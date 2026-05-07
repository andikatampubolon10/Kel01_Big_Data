from pyspark.sql import DataFrame
from pyspark.sql.functions import col, sum as fsum, countDistinct, max as fmax, datediff, lit

def add_total_amount(df_clean: DataFrame) -> DataFrame:
    """
    Tambah kolom TotalAmount = Quantity * UnitPrice.
    """
    return df_clean.withColumn("TotalAmount", col("Quantity") * col("UnitPrice"))

def compute_rfm(df_sales: DataFrame, reference_date=None) -> DataFrame:
    """
    Hitung RFM per CustomerID.
    - Recency: selisih hari dari pembelian terakhir terhadap reference_date
    - Frequency: jumlah invoice unik
    - Monetary: total spending

    reference_date:
      - None -> pakai max InvoiceTS dari dataset sebagai acuan
      - atau bisa diberi string '2011-12-10' dsb
    """
    if reference_date is None:
        ref = df_sales.select(fmax(col("InvoiceTS")).alias("ref_ts")).collect()[0]["ref_ts"]
        reference_date = ref

    # RFM agregasi
    rfm = (
        df_sales.groupBy("CustomerID")
        .agg(
            fmax(col("InvoiceTS")).alias("LastPurchaseTS"),
            countDistinct(col("InvoiceNo")).alias("Frequency"),
            fsum(col("TotalAmount")).alias("Monetary"),
        )
        .withColumn("Recency", datediff(lit(reference_date), col("LastPurchaseTS")))
    )
    return rfm