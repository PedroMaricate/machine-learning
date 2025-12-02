from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, when

# 1. Criar SparkSession
spark = (
    SparkSession.builder
    .appName("MBA-PySpark-Exercicio")
    .getOrCreate()
)

# 2. Ler base MBA
df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv("data/MBA.csv")
)

print("Esquema:")
df.printSchema()

print("Primeiras linhas:")
df.show(5)

# 3. Análises

# Total de candidatos
total = df.count()
print(f"Total de linhas: {total}")

# GMAT médio por major
gmat_major = (
    df.groupBy("major")
      .agg(avg("gmat").alias("gmat_medio"))
      .orderBy(col("gmat_medio").desc())
)

gmat_major.show()

# Taxa de admissão por gênero
df2 = df.withColumn(
    "admit_flag",
    when(col("admission") == "Admit", 1).otherwise(0)
)

taxa_genero = (
    df2.groupBy("gender")
       .agg(
           avg("admit_flag").alias("taxa_admissao"),
           count("*").alias("n")
       )
)

taxa_genero.show()

# 4. Salvar saídas
gmat_major.coalesce(1).write.mode("overwrite").option("header", True).csv("data/out_gmat_major")
taxa_genero.coalesce(1).write.mode("overwrite").option("header", True).csv("data/out_taxa_genero")

# 5. Gerar gráfico
pdf = gmat_major.toPandas()
import matplotlib.pyplot as plt

plt.figure(figsize=(8,4))
plt.bar(pdf["major"], pdf["gmat_medio"])
plt.xticks(rotation=45, ha="right")
plt.title("GMAT Médio por Major")
plt.tight_layout()
plt.savefig("data/grafico_gmat_major.png")

spark.stop()
