import os
import sys

# Set PySpark to use the current Python executable
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, upper, lit, when
from pyspark.ml.feature import MinMaxScaler, StandardScaler, StringIndexer, VectorAssembler
from pyspark.sql.types import DoubleType

def run_advanced_preprocessing():
    spark = SparkSession.builder \
        .appName("AdvancedPreprocessing") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

    # Path data
    input_path = "dataset/online_retail.csv"
    output_path = "dataset/preprocessed_data"

    print("1. Membaca dataset utama...")
    df = spark.read.csv(input_path, header=True, inferSchema=True)
    
    total_rows = df.count()
    print(f"Total baris awal: {total_rows}")

    # ==========================================
    # A. DATA CLEANING
    # ==========================================
    print("\n--- A. DATA CLEANING ---")
    
    # 1. Menangani Missing Value
    # Menghitung baris dengan missing value di kolom penting
    df_clean = df.dropna(subset=["Description", "CustomerID"])
    clean_rows = df_clean.count()
    missing_dropped = total_rows - clean_rows
    print(f"Baris dihapus karena missing value: {missing_dropped} ({missing_dropped/total_rows*100:.2f}%)")
    
    # 2. Mengatasi Data Tidak Konsisten
    # Contoh: Membuat format nama negara menjadi huruf kapital semua (Uppercase) agar konsisten
    # dan membuang Quantity <= 0
    df_clean = df_clean.withColumn("Country", upper(col("Country")))
    df_clean = df_clean.filter(col("Quantity") > 0)
    
    print(f"Total baris setelah cleaning konsistensi: {df_clean.count()}")

    # ==========================================
    # B. DATA INTEGRATION
    # ==========================================
    print("\n--- B. DATA INTEGRATION ---")
    # Mensimulasikan penggabungan dengan sumber data lain (misal tabel region negara)
    region_data = [
        ("UNITED KINGDOM", "Europe"),
        ("GERMANY", "Europe"),
        ("FRANCE", "Europe"),
        ("AUSTRALIA", "Oceania"),
        ("JAPAN", "Asia")
    ]
    df_region = spark.createDataFrame(region_data, ["Country", "Region"])
    
    # Left join dataset utama dengan dataset region
    df_integrated = df_clean.join(df_region, on="Country", how="left")
    
    # Isi null region dengan 'Other'
    df_integrated = df_integrated.fillna({"Region": "Other"})
    print("Berhasil menggabungkan (Integrasi) dengan data Region.")

    # ==========================================
    # C. FEATURE ENGINEERING
    # ==========================================
    print("\n--- C. FEATURE ENGINEERING ---")
    # Membuat fitur baru: TotalAmount (Pendapatan per baris)
    df_engineered = df_integrated.withColumn("TotalAmount", col("Quantity") * col("UnitPrice"))
    print("Fitur 'TotalAmount' berhasil dibuat.")

    # ==========================================
    # D. DATA TRANSFORMATION & ENCODING
    # ==========================================
    print("\n--- D. DATA TRANSFORMATION & ENCODING ---")
    
    # Pastikan tipe data numerik adalah Double untuk proses scaling
    df_engineered = df_engineered.withColumn("Quantity", col("Quantity").cast(DoubleType()))
    df_engineered = df_engineered.withColumn("UnitPrice", col("UnitPrice").cast(DoubleType()))
    df_engineered = df_engineered.withColumn("TotalAmount", col("TotalAmount").cast(DoubleType()))

    # 1. ENCODING (Mengubah teks menjadi angka)
    # Kita encode kolom 'Region' teks menjadi angka (Region_Index)
    indexer = StringIndexer(inputCol="Region", outputCol="Region_Encoded")
    df_encoded = indexer.fit(df_engineered).transform(df_engineered)
    print("Encoding berhasil: Kolom 'Region' diubah menjadi 'Region_Encoded'.")

    # VectorAssembler diperlukan Spark ML sebelum scaling
    vec_assembler_qty = VectorAssembler(inputCols=["Quantity"], outputCol="Quantity_Vec")
    vec_assembler_price = VectorAssembler(inputCols=["UnitPrice"], outputCol="UnitPrice_Vec")
    
    df_vec = vec_assembler_qty.transform(df_encoded)
    df_vec = vec_assembler_price.transform(df_vec)

    # 2. NORMALIZATION (Min-Max) pada Quantity
    # Menyamakan rentang data Quantity menjadi 0 sampai 1
    min_max_scaler = MinMaxScaler(inputCol="Quantity_Vec", outputCol="Quantity_Normalized")
    df_scaled = min_max_scaler.fit(df_vec).transform(df_vec)
    print("Normalisasi (Min-Max) berhasil diterapkan pada kolom 'Quantity'.")

    # 3. STANDARDIZATION (Z-Score) pada UnitPrice
    # Mengubah data UnitPrice menjadi distribusi dengan mean 0 dan standard deviasi 1
    std_scaler = StandardScaler(inputCol="UnitPrice_Vec", outputCol="UnitPrice_Standardized", withMean=True, withStd=True)
    df_scaled = std_scaler.fit(df_scaled).transform(df_scaled)
    print("Standardisasi (Z-Score) berhasil diterapkan pada kolom 'UnitPrice'.")

    # ==========================================
    # E. MENYIMPAN KE CSV BARU
    # ==========================================
    print("\n--- E. MENYIMPAN HASIL KE CSV ---")
    # Kita buang kolom vector sementara agar bisa disimpan ke CSV
    final_cols = [
        "InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate",
        "UnitPrice", "CustomerID", "Country", "Region", "Region_Encoded", "TotalAmount"
    ]
    # Spark tidak bisa menyimpan kolom tipe Vector langsung ke CSV, jadi kita ekstrak elemen pertamanya
    # Untuk kesederhanaan, kita hanya menyimpan fitur utama dan fitur engineered
    df_final = df_scaled.select(*final_cols)
    
    # Simpan ke format CSV (karena dataset besar, Spark akan menyimpannya dalam partisi/folder)
    # Kita menggunakan coalesce(1) agar outputnya menjadi 1 file CSV saja (untuk mempermudah).
    try:
        df_final.coalesce(1).write.csv(output_path, header=True, mode="overwrite")
        print(f"Sukses! Dataset hasil preprocessing telah disimpan ke folder: {output_path}")
    except Exception as e:
        print(f"Gagal menyimpan CSV: {e}")

    spark.stop()

if __name__ == "__main__":
    run_advanced_preprocessing()
