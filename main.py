import pandas as pd
import sklearn.model_selection as ms
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
import numpy as np


# Importando o banco e dividindo as colunas
dataset = pd.read_csv("db_test.csv")
dataset = dataset.dropna()
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# Codificando variaveis Dummy
X_dummies = pd.get_dummies(X)

# Separando dados em treino e teste
X_train, X_test, y_train, y_test = ms.train_test_split(X_dummies, y, test_size=1/5, random_state=0)

# Treinando o modelo
regressor = lm.LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

np.set_printoptions(precision=2)
result = np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.values.reshape(-1, 1)), 1)

X_reverse = X_test

X_reverse = X_reverse.reset_index(drop=True)

y_compare = pd.DataFrame(result)
y_compare = y_compare.rename(index=str, columns={0:'y_pred', 1:'y_test'})
y_compare = y_compare.reset_index(drop=True)

result_final = pd.concat([y_compare, X_reverse], axis=1)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)

