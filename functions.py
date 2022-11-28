from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, QuantileRegressor


def get_stats(y_test, y_pred):
    model_stats = []
    plt.scatter(y_test, y_pred)
    model_stats.append(metrics.r2_score(y_test, y_pred))
    model_stats.append(metrics.mean_absolute_percentage_error(y_test, y_pred))
    model_stats.append(metrics.mean_absolute_error(y_test, y_pred))
    model_stats.append(metrics.mean_squared_error(y_test, y_pred, squared=False))
    model_stats.append(metrics.mean_squared_error(y_test, y_pred, squared=True))
    return model_stats

def show_stats(model_accuracy, title):
    count = 0
    print(title)
    for i in model_accuracy:
        print("Fold ", count,
              "R²: %.4f" %i[0],
              "\t| MAPE: %.2f" %(i[1]*100), "%",
              "\t| MAE: %.2f" %i[2],
              "\t| MSE: %.2f" %i[3],
              "\t| RMSE: %.2f" %i[4])
        count += 1

def get_folder_media(model_stats, n):
    media = 0
    for i in model_stats:
        media += i[1] / n
    return media

def get_media(model_medias):
    media = 0
    for i in model_medias:
        media += i / len(model_medias)
    return media

def start_models(X_dummies, y, kfold, n):
    Linear_accuracy, Ridge_accuracy, Lasso_accuracy, Quantile_accuracy = [], [], [], []
    for train_index, test_index in kfold.split(X_dummies):
        #print("TRAIN:", train_index, "\nTEST:", test_index)
        X_train, X_test = X_dummies.iloc[train_index], X_dummies.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        linearR = LinearRegression().fit(X_train, y_train)
        ridgeR = Ridge(alpha=1.0, max_iter=10000).fit(X_train, y_train)
        lassoR = Lasso(alpha=1.0, max_iter=10000).fit(X_train, y_train)
        quantileR = QuantileRegressor().fit(X_train, y_train)
        
        linear_pred = linearR.predict(X_test)
        ridge_pred = ridgeR.predict(X_test)
        lasso_pred = lassoR.predict(X_test)
        quantile_pred = quantileR.predict(X_test)
        
        Linear_accuracy.append(get_stats(y_test, linear_pred))
        Ridge_accuracy.append(get_stats(y_test, ridge_pred))
        Lasso_accuracy.append(get_stats(y_test, lasso_pred))
        Quantile_accuracy.append(get_stats(y_test, quantile_pred))
    
    return Linear_accuracy, Ridge_accuracy, Lasso_accuracy, Quantile_accuracy