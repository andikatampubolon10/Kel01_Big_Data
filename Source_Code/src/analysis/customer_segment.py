from pyspark.sql import DataFrame
from pyspark.sql.functions import col, when
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

def rfm_kmeans(rfm: DataFrame, k: int = 5):
    """
    Jalankan KMeans pada fitur (Recency, Frequency, Monetary).
    Return: (df_with_cluster, silhouette)
    """
    # Memasukkan fitur tambahan hasil preprocessing (Country & Qty) ke dalam KMeans
    features = ["Recency", "Frequency", "Monetary", "Country_Encoded", "Sum_Qty"]
    df = rfm.select("CustomerID", *features).na.drop()

    assembler = VectorAssembler(inputCols=features, outputCol="features_raw")
    assembled = assembler.transform(df)

    scaler = StandardScaler(inputCol="features_raw", outputCol="features", withMean=True, withStd=True)
    scaled_model = scaler.fit(assembled)
    scaled = scaled_model.transform(assembled)

    # 4) Inisialisasi centroid dengan k-means++ (k-means|| di Spark) & 5) Jalankan iterasi
    kmeans = KMeans(k=k, seed=42, featuresCol="features", predictionCol="cluster", initMode="k-means||", maxIter=300, tol=1e-4)
    model = kmeans.fit(scaled)
    pred = model.transform(scaled)

    # 6) Evaluasi hasil clustering (Silhouette Score & SSE/Inertia)
    evaluator = ClusteringEvaluator(featuresCol="features", predictionCol="cluster", metricName="silhouette", distanceMeasure="squaredEuclidean")
    silhouette = evaluator.evaluate(pred)
    
    # SSE (Sum of Squared Errors)
    sse = model.summary.trainingCost
    print(f"Evaluasi K-Means (K={k}) -> Silhouette Score: {silhouette:.4f} | SSE (Inertia): {sse:.4f}")

    return pred.select("CustomerID", "Recency", "Frequency", "Monetary", "Country_Encoded", "Sum_Qty", "cluster"), silhouette

def label_clusters_simple(df_clustered: DataFrame) -> DataFrame:
    """
    Label sederhana berbasis cluster id.
    Untuk laporan, sebaiknya label ditentukan setelah lihat statistik per cluster.
    """
    return (
        df_clustered
        .withColumn(
            "segment_label",
            when(col("cluster") == 0, "Segment-0")
            .when(col("cluster") == 1, "Segment-1")
            .when(col("cluster") == 2, "Segment-2")
            .when(col("cluster") == 3, "Segment-3")
            .when(col("cluster") == 4, "Segment-4")
            .otherwise("Segment-Other")
        )
    )