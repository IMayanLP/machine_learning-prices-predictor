import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from functions import start_models, show_stats, get_folder_media, get_media

np.set_printoptions(precision=2)

db = pd.read_csv("banco_v9.csv")
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
reps = 10

print("Num de Folders: ", kfold.get_n_splits(X_dummies))

ridge_medias, linear_medias, lasso_medias, quantile_medias = [], [], [], []
for i in range(reps):
    Linear_accuracy, Ridge_accuracy, Lasso_accuracy, Quantile_accuracy = start_models(X_dummies, y, kfold, reps)
    show_stats(Linear_accuracy, "Linear Regression")
    show_stats(Ridge_accuracy, "Ridge Regression")
    show_stats(Lasso_accuracy, "Lasso Regression")
    show_stats(Quantile_accuracy, "Quantile Regression")
    linear_medias.append(get_folder_media(Linear_accuracy, reps))
    ridge_medias.append(get_folder_media(Ridge_accuracy, reps))
    lasso_medias.append(get_folder_media(Lasso_accuracy, reps))
    quantile_medias.append(get_folder_media(Quantile_accuracy, reps))
    print("\n")

print("\nMedias em ", reps, "repetições:",
      "\nLinear: %.2f" % (get_media(linear_medias) * 100), "%",
      "\nRidge: %.2f" % (get_media(ridge_medias)  * 100), "%",
      "\nLasso: %.2f" % (get_media(lasso_medias)  * 100), "%",
      "\nQuantile: %.2f" % (get_media(quantile_medias)  * 100), "%",)