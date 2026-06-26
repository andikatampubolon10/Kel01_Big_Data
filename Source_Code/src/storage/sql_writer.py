from pyspark.sql import DataFrame

def write_df_to_postgres(df: DataFrame, table: str, jdbc_url: str, jdbc_props: dict, mode: str = "overwrite"):
    """
    Simpan dataframe ke PostgreSQL via JDBC.
    mode: overwrite / append
    """
    try:
        (
            df.write
            .mode(mode)
            .jdbc(url=jdbc_url, table=table, properties=jdbc_props)
        )
        print(f"Sukses menyimpan ke PostgreSQL (tabel: {table})")
    except Exception as e:
        print(f"\n⚠️ GAGAL MENYIMPAN KE POSTGRESQL (Network/DB Error) ⚠️")
        print(f"Pesan Error: {e}")
        print("Pipeline akan dilanjutkan secara offline (dataset CSV sudah dibuat dengan sukses).")

def read_df_from_postgres(spark, table: str, jdbc_url: str, jdbc_props: dict) -> DataFrame:
    return spark.read.jdbc(url=jdbc_url, table=table, properties=jdbc_props)