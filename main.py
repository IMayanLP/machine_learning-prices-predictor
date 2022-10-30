import pandas as pd
import sklearn.model_selection as ms
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
import numpy as np


# Importando o banco e dividindo as colunas
dataset = pd.read_csv("db_test.csv") # Carrega o banco
dataset = dataset.dropna() # Exclui os nulos
X = dataset.iloc[:, :-1] # Pega todas as colunas menos a ultima
y = dataset.iloc[:, -1] # Pega apenas a ultima coluna

# Codificando variaveis Dummy
X_dummies = pd.get_dummies(X) # Classificando variaveis não quantificativas (Dummies)

# Separando dados em treino e teste
X_train, X_test, y_train, y_test = ms.train_test_split(X_dummies, y, test_size=1/7, random_state=0)

# Treinando o modelo
regressor = lm.LinearRegression()
regressor.fit(X_train, y_train)

# Criando conjunto de predict a partir do modelo treinado
y_pred = regressor.predict(X_test) # Prevendo valores de X_test

np.set_printoptions(precision=2) # Configurando precisão dos numeros
# Concatenando o conjunto de predict com o conjunto de teste em colunas
result = np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.values.reshape(-1, 1)), 1)

# Re-ordenando o conjunto de teste
X_test = X_test.reset_index(drop=True)

# Criando conjuntos com os dados de result
y_compare = pd.DataFrame(result)
# Renomeando colunas
y_compare = y_compare.rename(index=str, columns={0:'y_pred', 1:'y_test'})
# Re-ordenando o conjunto y_compare
y_compare = y_compare.reset_index(drop=True)

# Concatenando as colunas da comparação e testes
result_final = pd.concat([y_compare, X_test], axis=1)

# Criando R²
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)

