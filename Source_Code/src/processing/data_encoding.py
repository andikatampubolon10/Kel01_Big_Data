from pyspark.sql import DataFrame
from pyspark.ml.feature import StringIndexer

def encode_data(df: DataFrame) -> DataFrame:
    """
    Mengubah variabel kategorikal teks menjadi numerik (Country).
    """
    indexer = StringIndexer(inputCol="Country", outputCol="Country_Encoded", handleInvalid="keep", stringOrderType="frequencyDesc")
    df_encoded = indexer.fit(df).transform(df)
    return df_encoded
