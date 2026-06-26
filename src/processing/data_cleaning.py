from pyspark.sql import DataFrame
from pyspark.sql.functions import col, upper, trim, to_timestamp

def clean_data(df: DataFrame) -> DataFrame:
    """
    Melakukan proses pembersihan data:
    1. Menghapus missing values pada CustomerID & Description
    2. Konsistensi data: trim spasi, kapitalisasi Country
    3. Konversi tipe data tanggal
    4. Hapus nilai Quantity/UnitPrice yang minus atau nol
    """
    # 1. Drop Missing Value
    df_clean = df.dropna(subset=["Description", "CustomerID"])
    
    # 2. Konsistensi Data & Trim
    for c in ["InvoiceNo", "StockCode", "Description", "CustomerID", "Country", "InvoiceDate"]:
        df_clean = df_clean.withColumn(c, trim(col(c)))
        
    df_clean = df_clean.withColumn("Country", upper(col("Country")))
    
    # 3. Konversi format waktu
    df_clean = df_clean.withColumn("InvoiceTS", to_timestamp(col("InvoiceDate"), "yyyy-MM-dd HH:mm:ss"))
    df_clean = df_clean.filter(col("InvoiceTS").isNotNull())
    
    # 4. Hapus anomali (nilai negatif & transaksi batal)
    df_clean = df_clean.filter(~col("InvoiceNo").startswith("C"))
    df_clean = df_clean.filter((col("Quantity") > 0) & (col("UnitPrice") > 0))
    
    return df_clean
