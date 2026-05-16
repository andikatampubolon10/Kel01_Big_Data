import os
import sys
from dotenv import load_dotenv

# Agar import config/ dan src/ bisa jalan saat run dari root
sys.path.append(os.getcwd())

load_dotenv()

from config.spark_config import create_spark
from config.db_config import PG_JDBC_URL, PG_JDBC_PROPS, MONGO_URI, MONGO_DB

from src.ingestion.load_csv_to_spark import read_online_retail_csv
from src.processing.cleaning import clean_online_retail
from src.processing.transformations import add_total_amount, compute_rfm

from src.analysis.sales_trend import sales_monthly
from src.analysis.customer_segment import rfm_kmeans, label_clusters_simple

from src.storage.sql_writer import write_df_to_postgres
from src.storage.nosql_writer import upsert_customer_segments


def main():
    spark = create_spark()

    # Path dataset kamu adalah dataset/, bukan data/
    df_raw = read_online_retail_csv(spark, "dataset/online_retail.csv")
    print("Rows df_raw:", df_raw.count())
    df_clean = clean_online_retail(df_raw, drop_null_customer=True)
    df_sales = add_total_amount(df_clean)

    # --- Write monthly sales aggregation to Supabase/Postgres
    df_monthly = sales_monthly(df_sales)
    write_df_to_postgres(
        df_monthly,
        table="agg_sales_monthly",
        jdbc_url=PG_JDBC_URL,
        jdbc_props=PG_JDBC_PROPS,
        mode="overwrite",
    )

    # --- RFM + KMeans
    rfm = compute_rfm(df_sales)
    df_clustered, sil = rfm_kmeans(rfm, k=4)
    df_labeled = label_clusters_simple(df_clustered)
    print("Silhouette:", sil)

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

    print("Rows df_labeled:", df_labeled.count())
    print("Clean:", df_clean.count())
    print("Sales:", df_sales.count())

    upsert_customer_segments(docs, MONGO_URI, MONGO_DB, collection="customer_segments")

    spark.stop()


if __name__ == "__main__":
    main()