import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("./src/MBA.csv")

print(df.sample(frac=.01).to_markdown(index=False))