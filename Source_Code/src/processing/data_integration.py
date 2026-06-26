from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, when

def integrate_data(df: DataFrame, spark: SparkSession) -> DataFrame:
    """
    Menggabungkan dataset transaksi dengan sumber data eksternal (mapping benua/Region)
    menggunakan native Spark SQL functions untuk menghindari overhead Python Worker di Windows.
    """
    df_integrated = df.withColumn(
        "Region",
        when(col("Country") == "UNITED KINGDOM", "Europe")
        .when(col("Country") == "GERMANY", "Europe")
        .when(col("Country") == "FRANCE", "Europe")
        .when(col("Country") == "AUSTRALIA", "Oceania")
        .when(col("Country") == "JAPAN", "Asia")
        .otherwise("Other")
    )
    
    return df_integrated
