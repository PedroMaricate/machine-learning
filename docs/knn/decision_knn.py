import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("./src/MBA.csv")

df["admission"] = df["admission"].fillna("Deny")
df["race"] = df["race"].fillna("Unknown")

adm_map = {"Deny": 0, "Waitlist": 1, "Admit": 2}
y = df["admission"].map(adm_map).astype(int)

num_cols = ["gpa", "gmat", "work_exp"]
cat_cols = ["gender", "international", "major", "race", "work_industry"]

for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
    df[c] = df[c].fillna(df[c].median())

X_cat = pd.get_dummies(df[cat_cols].astype(str).apply(lambda s: s.str.strip()),
                       drop_first=False, dtype=int)
X_num = df[num_cols].copy()

scaler = StandardScaler()
X_num[num_cols] = scaler.fit_transform(X_num[num_cols])

X = pd.concat([X_num, X_cat], axis=1).values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pca = PCA(n_components=2)
X_train_2d = pca.fit_transform(X_train)
X_test_2d = pca.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_2d, y_train)
predictions = knn.predict(X_test_2d)
print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")

plt.figure(figsize=(12, 10))
h = 0.05
x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.3)
sns.scatterplot(x=X_train_2d[:, 0], y=X_train_2d[:, 1], hue=y_train,
                palette="deep", s=100, edgecolor="k", alpha=0.8)

plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("KNN Decision Boundary (MBA Dataset)")

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())
