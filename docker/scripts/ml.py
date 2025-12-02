from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("PySpark Tutorial Example")
    .getOrCreate()
)

df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv("data/sample.csv")
)

df.show()
spark.stop()
