from pyspark.sql import DataFrame
import pandas as pd
import os

def export_to_csv(df: DataFrame, output_path: str):
    """
    Menyimpan hasil dataframe ke format CSV menggunakan Pandas 
    (untuk membypass error Hadoop NativeIO di Windows).
    """
    final_cols = [
        "InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "InvoiceTS",
        "UnitPrice", "CustomerID", "Country", "Country_Encoded", "Region", "TotalAmount",
        "Quantity_Normalized", "UnitPrice_Standardized"
    ]
    df_final = df.select(*final_cols)
    
    print("Mengkonversi Spark DataFrame ke Pandas untuk ekspor (menghindari error Hadoop di Windows)...")
    pdf = df_final.toPandas()
    
    os.makedirs(output_path, exist_ok=True)
    file_path = os.path.join(output_path, "preprocessed_data.csv")
    pdf.to_csv(file_path, index=False)
    
    print(f"Data sukses diekspor ke file tunggal: {file_path}")
