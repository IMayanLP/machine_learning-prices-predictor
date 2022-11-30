import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from functions import start_models, show_stats, get_folder_media, get_media

np.set_printoptions(precision=2)

db = pd.read_csv("banco_final.csv")
db = db.dropna()

"""
db_normalized = db.copy()

db_normalized['Processador (score)'] = db_normalized['Processador (score)'] / db['Processador (score)'].max()
db_normalized['Placa de video (score)'] = db_normalized['Placa de video (score)'] / db['Placa de video (score)'].max()
"""

X = db.iloc[:, :-1]
y = db.iloc[:, -1]

X_dummies = pd.get_dummies(X)

kfold  = KFold(n_splits=10, shuffle=True)
reps = 100

print("Num de Folders: ", kfold.get_n_splits(X_dummies))

ridge_medias, linear_medias, lasso_medias, quantile_medias = [], [], [], []
medias_gerais = []

for i in range(reps):
    print(i+1, "º repetição")
    medias_linha = []
    Linear_accuracy, Ridge_accuracy, Lasso_accuracy, Quantile_accuracy = start_models(X_dummies, y, kfold, reps)
    show_stats(Linear_accuracy, "Linear Regression")
    show_stats(Ridge_accuracy, "Ridge Regression")
    show_stats(Lasso_accuracy, "Lasso Regression")
    show_stats(Quantile_accuracy, "Quantile Regression")
    linear_m, ridge_m, lasso_m, quantile_m = get_folder_media(Linear_accuracy), get_folder_media(Ridge_accuracy), get_folder_media(Lasso_accuracy), get_folder_media(Quantile_accuracy)
    linear_medias.append(linear_m)
    ridge_medias.append(ridge_m)
    lasso_medias.append(lasso_m)
    quantile_medias.append(quantile_m)
    
    medias_linha.append(linear_m)
    medias_linha.append(ridge_m)
    medias_linha.append(lasso_m)
    medias_linha.append(quantile_m)
    medias_gerais.append(medias_linha)
    print("\n")

print("\nMedias em ", reps, "repetições:",
      "\nLinear: %.5f" % (get_media(linear_medias) * 100), "%",
      "\nRidge: %.5f" % (get_media(ridge_medias)  * 100), "%",
      "\nLasso: %.5f" % (get_media(lasso_medias)  * 100), "%",
      "\nQuantile: %.5f" % (get_media(quantile_medias)  * 100), "%")

counts = [0, 0, 0, 0]
for i in range(reps):
    melhor = medias_gerais[i][0]
    indice = 0
    for j in range(len(counts)):
        if medias_gerais[i][j] < melhor:
            melhor = medias_gerais[i][j]
            indice = j
    counts[indice] += 1

print("\n", counts)