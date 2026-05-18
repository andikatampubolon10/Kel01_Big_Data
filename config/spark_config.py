from pyspark.sql import SparkSession

def create_spark(app_name: str = "Proyek_BigData_Retail"):
    """
    Buat SparkSession dengan optimasi memori untuk laptop.
    """
    spark = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.jars.packages", "org.postgresql:postgresql:42.7.3")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "10")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark