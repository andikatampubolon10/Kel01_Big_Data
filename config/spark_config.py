from pyspark.sql import SparkSession

def create_spark(app_name: str = "Proyek_BigData_Retail"):
    """
    Buat SparkSession.
    Catatan:
    - Untuk koneksi DB via JDBC dan Mongo Spark Connector, biasanya butuh JAR/connector.
    - Jika environment kamu tidak bisa download packages otomatis, siapkan jar secara manual.
    """
    spark = (
        SparkSession.builder
        .appName(app_name)
        # Uncomment jika mau atur memori (sesuaikan laptop)
        # .config("spark.driver.memory", "4g")
        # .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark