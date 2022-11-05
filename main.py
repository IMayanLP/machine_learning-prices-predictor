import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score # função de validação cruzada
from sklearn.model_selection import KFold # KFold, só isso mesmo.
from sklearn.linear_model import LinearRegression # função de regressão linear

# Importando o banco e dividindo as colunas
db = pd.read_csv("db_test.csv") # Carrega o banco
db = db.dropna() # Exclui os nulos
X = db.iloc[:, :-1] # Pega todas as colunas menos a ultima
y = db.iloc[:, -1] # Pega apenas a ultima coluna

# Codificando variaveis Dummy
X_dummies = pd.get_dummies(X) # Classificando variaveis não quantificativas (Dummies)
# inclui variáveis categóricas ao modelo (que só aceita variáveis numéricas)

modelo  = LinearRegression() # criando modelo de regressão linear
# configurando o K-Fold com K = 10 e embaralhamento ligado
kfold  = KFold(n_splits=10, shuffle=True)
# chama a função de validação cruzada com nosso modelo, X, y
# e determina a estratégia de divisão de validação cruzada (no caso kfold declarado acima)
result = cross_val_score(modelo, X_dummies, y, cv = kfold)

print("pontuações R² de cada K-Fold: {0}".format(result))
print("Média dos R²: {0}".format(result.mean()))


