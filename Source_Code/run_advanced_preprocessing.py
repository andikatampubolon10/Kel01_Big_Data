import os
import sys

# Set PySpark to use the current Python executable
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

from pyspark.sql import SparkSession
from src.processing.data_cleaning import clean_data
from src.processing.data_integration import integrate_data
from src.processing.feature_engineering import create_features
from src.processing.data_encoding import encode_data
from src.processing.data_transformation import transform_data
from src.processing.export_data import export_to_csv

def main():
    spark = SparkSession.builder \
        .appName("ModularPreprocessing") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

    input_path = "dataset/online_retail.csv"
    output_path = "dataset/preprocessed_data_modular"

    print("=== MULAI PROSES PREPROCESSING MODULAR ===")
    
    # 0. Load Data
    print("Membaca dataset mentah...")
    df_raw = spark.read.csv(input_path, header=True, inferSchema=True)
    
    # 1. Data Cleaning
    print("1. Menjalankan Data Cleaning...")
    df_clean = clean_data(df_raw)
    
    # 2. Data Integration
    print("2. Menjalankan Data Integration...")
    df_integrated = integrate_data(df_clean, spark)
    
    # 3. Feature Engineering
    print("3. Menjalankan Feature Engineering...")
    df_featured = create_features(df_integrated)
    
    # 4. Data Encoding
    print("4. Menjalankan Data Encoding...")
    df_encoded = encode_data(df_featured)
    
    # 5. Data Transformation (Norm & Std)
    print("5. Menjalankan Data Transformation...")
    df_transformed = transform_data(df_encoded)
    
    # 6. Export Data
    print("6. Menyimpan Hasil (Export to CSV)...")
    export_to_csv(df_transformed, output_path)

    print("=== PROSES SELESAI ===")
    spark.stop()

if __name__ == "__main__":
    main()
