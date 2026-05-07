from pyspark.sql import DataFrame
from pyspark.sql.functions import col, when
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

def rfm_kmeans(rfm: DataFrame, k: int = 4):
    """
    Jalankan KMeans pada fitur (Recency, Frequency, Monetary).
    Return: (df_with_cluster, silhouette)
    """
    features = ["Recency", "Frequency", "Monetary"]
    df = rfm.select("CustomerID", *features).na.drop()

    assembler = VectorAssembler(inputCols=features, outputCol="features_raw")
    assembled = assembler.transform(df)

    scaler = StandardScaler(inputCol="features_raw", outputCol="features", withMean=True, withStd=True)
    scaled_model = scaler.fit(assembled)
    scaled = scaled_model.transform(assembled)

    kmeans = KMeans(k=k, seed=42, featuresCol="features", predictionCol="cluster")
    model = kmeans.fit(scaled)
    pred = model.transform(scaled)

    evaluator = ClusteringEvaluator(featuresCol="features", predictionCol="cluster", metricName="silhouette", distanceMeasure="squaredEuclidean")
    silhouette = evaluator.evaluate(pred)

    return pred.select("CustomerID", "Recency", "Frequency", "Monetary", "cluster"), silhouette

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
            .otherwise("Segment-Other")
        )
    )