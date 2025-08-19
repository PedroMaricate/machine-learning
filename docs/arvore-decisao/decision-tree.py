import matplotlib.pyplot as plt
import pandas as pd

def preprocess(df):
    # Fill missing values
    df['admission'].fillna(df['admission'].mode()[0], inplace=True)
    # Select features
    features = ['gender', 'international', 'gpa', 'major', 'race', 'gmat', 'work_exp', 'work_industry', 'admission']
    return df[features]

# Load the dataset
df = pd.read_csv('C:/Users/pedro.maricate/Downloads/baseMBA/MBA.csv')

df = df.sample(n=10)
# Preprocessing
df = preprocess(df)

# Display the first few rows of the dataset
print(df)


