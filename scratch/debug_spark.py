import os
import sys

# Workaround for 'JavaPackage' object is not callable
if 'SPARK_HOME' in os.environ:
    print(f"Unsetting SPARK_HOME: {os.environ['SPARK_HOME']}")
    del os.environ['SPARK_HOME']

from pyspark.sql import SparkSession

try:
    spark = SparkSession.builder.appName("TestFix").getOrCreate()
    print(f"Spark Version: {spark.version}")
    spark.stop()
    print("SUCCESS!")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
