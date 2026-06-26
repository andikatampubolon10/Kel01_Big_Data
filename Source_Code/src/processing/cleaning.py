from pyspark.sql import DataFrame
from pyspark.sql.functions import col, trim, to_timestamp

def clean_online_retail(df_raw: DataFrame, drop_null_customer: bool = True) -> DataFrame:
    """
    Cleaning wajib:
    - Trim string
    - Parse InvoiceDate -> InvoiceTS (timestamp)
    - Filter transaksi cancel (InvoiceNo diawali 'C')
    - Filter Quantity dan UnitPrice > 0 (sales bersih)
    - Drop CustomerID null (untuk segmentasi)
    """
    df = df_raw

    # trim kolom string utama
    for c in ["InvoiceNo", "StockCode", "Description", "CustomerID", "Country", "InvoiceDate"]:
        df = df.withColumn(c, trim(col(c)))

    # parse timestamp; format di CSV: "2010-12-01 08:26:00" (yyyy-MM-dd HH:mm:ss)
    df = df.withColumn("InvoiceTS", to_timestamp(col("InvoiceDate"), "yyyy-MM-dd HH:mm:ss"))

    # buang baris yang gagal parse tanggal
    df = df.filter(col("InvoiceTS").isNotNull())

    # buang transaksi cancel (InvoiceNo mulai C)
    df = df.filter(~col("InvoiceNo").startswith("C"))

    # filter nilai tidak masuk akal untuk penjualan bersih
    df = df.filter((col("Quantity") > 0) & (col("UnitPrice") > 0))

    if drop_null_customer:
        df = df.filter(col("CustomerID").isNotNull() & (col("CustomerID") != ""))

    # Description boleh null; kalau mau aman untuk analisis produk, bisa isi "UNKNOWN"
    # df = df.fillna({"Description": "UNKNOWN"})

    return df