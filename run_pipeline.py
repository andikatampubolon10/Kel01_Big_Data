import os
import sys
import argparse
from dotenv import load_dotenv

# Set PySpark to use the current Python executable (Mencegah error 'Python was not found' di Windows)
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# Agar import config/ dan src/ bisa jalan saat run dari root
sys.path.append(os.getcwd())

load_dotenv()

from config.spark_config import create_spark
from config.db_config import PG_JDBC_URL, PG_JDBC_PROPS, MONGO_URI, MONGO_DB

from src.ingestion.load_csv_to_spark import read_online_retail_csv
from src.processing.transformations import compute_rfm

# Import modular preprocessing
from src.processing.data_cleaning import clean_data
from src.processing.data_integration import integrate_data
from src.processing.feature_engineering import create_features
from src.processing.data_encoding import encode_data
from src.processing.data_transformation import transform_data
from src.processing.export_data import export_to_csv
from src.analysis.sales_trend import sales_monthly
from src.analysis.customer_segment import rfm_kmeans, label_clusters_simple

from src.storage.sql_writer import write_df_to_postgres
from src.storage.nosql_writer import upsert_customer_segments


def main(k_clusters: int = 5):
    spark = create_spark()

    # --- Step 1: Ingestion ---
    print(">>> Membaca data dari CSV...")
    df_raw = read_online_retail_csv(spark, "dataset/online_retail.csv")
    print(f"Rows df_raw: {df_raw.count()}")

    # --- Step 2: Advanced Preprocessing (Modular) ---
    print(">>> Melakukan data cleaning modular...")
    df_clean = clean_data(df_raw)
    
    print(">>> Melakukan data integration...")
    df_integrated = integrate_data(df_clean, spark)
    
    print(">>> Melakukan feature engineering...")
    df_featured = create_features(df_integrated)
    
    print(">>> Melakukan data encoding...")
    df_encoded = encode_data(df_featured)
    
    print(">>> Melakukan data transformation (Norm & Std)...")
    df_transformed = transform_data(df_encoded)
    
    print(">>> Menyimpan hasil data preprocessed ke CSV...")
    export_to_csv(df_transformed, "dataset/preprocessed_data_modular")
    
    # Set data yang telah diproses untuk tahap analisis lanjutan (RFM & Tren)
    df_sales = df_transformed
    print(f"Rows siap diproses (df_sales): {df_sales.count()}")

    # --- Step 3: Sales Trend (SQL) ---
    print(">>> Menghitung tren penjualan bulanan...")
    df_monthly = sales_monthly(df_sales)
    print(">>> Menyimpan tren ke PostgreSQL (Supabase)...")
    write_df_to_postgres(
        df_monthly,
        table="agg_sales_monthly",
        jdbc_url=PG_JDBC_URL,
        jdbc_props=PG_JDBC_PROPS,
        mode="overwrite",
    )

    # --- Step 4: RFM & Clustering (NoSQL) ---
    print(">>> Menghitung RFM...")
    rfm = compute_rfm(df_sales)
    print(f"RFM Count: {rfm.count()}")

    print(f">>> Menjalankan algoritma KMeans dengan K={k_clusters} (ini mungkin memakan waktu)...")
    df_clustered, sil = rfm_kmeans(rfm, k=k_clusters)
    df_labeled = label_clusters_simple(df_clustered)
    print(f"Silhouette Score: {sil}")

    # --- Write segments to MongoDB
    pdf = df_labeled.toPandas()  # aman karena jumlah customer jauh lebih kecil dari baris transaksi
    docs = []
    for _, r in pdf.iterrows():
        docs.append(
            {
                "_id": str(r["CustomerID"]),
                "rfm": {
                    "recency": int(r["Recency"]),
                    "frequency": int(r["Frequency"]),
                    "monetary": float(r["Monetary"]),
                },
                "segment": {
                    "cluster_id": int(r["cluster"]),
                    "label": str(r["segment_label"]),
                },
            }
        )

    print(">>> Menyimpan hasil segmentasi ke MongoDB Atlas...")
    upsert_customer_segments(docs, MONGO_URI, MONGO_DB, collection="customer_segments")

    print(">>> Pipeline Selesai dengan Sukses!")
    spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Big Data Analytics Pipeline")
    parser.add_argument("--k", type=int, default=5, help="Jumlah K (cluster) untuk K-Means")
    args = parser.parse_args()
    
    main(k_clusters=args.k)