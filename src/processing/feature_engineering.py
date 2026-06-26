from pyspark.sql import DataFrame
from pyspark.sql.functions import col

def create_features(df: DataFrame) -> DataFrame:
    """
    Membuat fitur/variabel turunan baru.
    """
    # Membuat TotalAmount
    df_featured = df.withColumn("TotalAmount", col("Quantity") * col("UnitPrice"))
    return df_featured
