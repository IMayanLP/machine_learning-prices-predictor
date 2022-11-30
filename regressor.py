import pandas as pd
from sklearn.linear_model import Ridge

class Regressor():
    def __init__(self):
        self.db = pd.read_csv("banco_final.csv")
        self.db = self.db.dropna()

        self.X = self.db.iloc[:, :-1]
        self.y = self.db.iloc[:, -1]

        self.X_dummies = pd.get_dummies(self.X)

        self.regressor = Ridge(alpha=1.0, max_iter=10000)
        self.regressor.fit(self.X_dummies, self.y)
    
    def predict(self, X):
        return self.regressor.predict(X)