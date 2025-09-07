import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("C:/Users/pedro.maricate/Downloads/baseMBA/MBA.csv")

print(df.sample(frac=.01).to_markdown(index=False))