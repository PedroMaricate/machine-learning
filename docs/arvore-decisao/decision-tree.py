import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from io import StringIO
from sklearn import tree
from sklearn.metrics import accuracy_score

def preprocess(df):
    
    # Fill missing values
    df["admission"] = df["admission"].fillna("Deny") 
    df['race'] = df['race'].fillna('Unknown')
    
    # Convert categorical variables
    label_encoder = LabelEncoder()
    df['gender'] = label_encoder.fit_transform(df['gender'])
    df['international'] = label_encoder.fit_transform(df['international'])
    df['major'] = label_encoder.fit_transform(df['major'])
    df['race'] = label_encoder.fit_transform(df['race'])
    df['work_industry'] = label_encoder.fit_transform(df['work_industry'])
    df['admission'] = label_encoder.fit_transform(df['admission'])
    
    # Select features
    features = ['gender', 'international', 'gpa', 'major', 'race', 'gmat', 'work_exp', 'work_industry', 'admission']
    return df[features]
     
# Load the dataset
df = pd.read_csv('C:/Users/pedro.maricate/Downloads/baseMBA/MBA.csv')

# Preprocessing
df = preprocess(df)

# Carregar o conjunto de dados
x = df.drop(columns=['admission'])
y = df['admission']

# Dividir os dados em conjuntos de treinamento e teste
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2,random_state=42)

# Criar e treinar o modelo de árvore de decisão
classifier = tree.DecisionTreeClassifier()
classifier.fit(x_train, y_train)

# Avaliar o modelo
accuracy = classifier.score(x_test, y_test)
print(f"Accuracy: {accuracy:.2f}")
tree.plot_tree(classifier)

feature_importance = pd.DataFrame({
    'Feature': classifier.feature_names_in_,
    'Importância': classifier.feature_importances_
})
print("<br>Importância das Features:")
print(feature_importance.sort_values(by='Importância', ascending=False).to_html())

plt.figure(figsize=(20, 10))
tree.plot_tree(classifier, max_depth=5, fontsize=10)


# Para imprimir na página HTML
buffer = StringIO()
plt.savefig(buffer, format="svg")
print(buffer.getvalue())




