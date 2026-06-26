from pyspark.sql import DataFrame
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import MinMaxScaler, StandardScaler, VectorAssembler
from pyspark.ml.functions import vector_to_array

def transform_data(df: DataFrame) -> DataFrame:
    """
    Menerapkan Normalisasi dan Standardisasi pada data.
    """
    # Cast tipe data
    df = df.withColumn("Quantity", col("Quantity").cast(DoubleType()))
    df = df.withColumn("UnitPrice", col("UnitPrice").cast(DoubleType()))

    # Vector Assembler
    vec_qty = VectorAssembler(inputCols=["Quantity"], outputCol="Quantity_Vec")
    vec_price = VectorAssembler(inputCols=["UnitPrice"], outputCol="UnitPrice_Vec")
    
    df_vec = vec_qty.transform(df)
    df_vec = vec_price.transform(df_vec)

    # 1. Min-Max Normalization (Quantity)
    min_max_scaler = MinMaxScaler(inputCol="Quantity_Vec", outputCol="Quantity_Normalized_Vec")
    df_scaled = min_max_scaler.fit(df_vec).transform(df_vec)

    # 2. Standardization (UnitPrice)
    std_scaler = StandardScaler(inputCol="UnitPrice_Vec", outputCol="UnitPrice_Standardized_Vec", withMean=True, withStd=True)
    df_scaled = std_scaler.fit(df_scaled).transform(df_scaled)
    
    # 3. Ekstrak nilai numerik dari Vector menggunakan fungsi native Spark (Mencegah Python Worker Crash)
    df_scaled = df_scaled.withColumn("Quantity_Normalized", vector_to_array(col("Quantity_Normalized_Vec")).getItem(0))
    df_scaled = df_scaled.withColumn("UnitPrice_Standardized", vector_to_array(col("UnitPrice_Standardized_Vec")).getItem(0))
    
    return df_scaled
