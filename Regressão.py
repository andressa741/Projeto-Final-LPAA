# %%
# Import as bibliotecas 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from pyparsing import col
import seaborn as sns
# %%
# Import do dataset
from sklearn.datasets import load_diabetes
X,y = load_diabetes(return_X_y = True, as_frame = True, scaled = False)
# %%
# Tratamento e preparação dos dados
print(X.head())
print(load_diabetes().DESCR)
print(X.info())
# %%
# Mudando o tipo da coluna sex para int64
X['sex'] = X['sex'].astype(np.int64)
print("Shape X:")
print(X.shape)
print("Shape y:")
print(y.shape)
print(X.describe())
 
# %%
sns.pairplot(X)
# %%
sns.heatmap(X.corr(),annot = True)
# %%
# Verificando outliers
plt.figure()
sns.boxplot(x="s1", data = X)
plt.show()
plt.figure()
sns.boxplot(x="s2", data = X)
plt.show()
plt.figure()
sns.boxplot(x="s3", data = X)
plt.show()
plt.figure()
sns.boxplot(x="s4", data = X)
plt.show()
plt.figure()
sns.boxplot(x="s5", data = X)
plt.show()
plt.figure()
sns.boxplot(x="s6", data = X)
plt.show()
# %%
# Separação entre treinamento e teste 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
print("Shape: X_train")
print(X_train.shape)
print("Shape: X_test")
print(X_test.shape)
print("Shape: y_train")
print(y_train.shape)
print("Shape: y_test")
print(y_test.shape)
# %%
# Feature Scaling

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
numeric_columns=list(X.select_dtypes('float64').columns)
categorical_columns=list(X.select_dtypes('int64').columns)
X_train = pd.get_dummies(X_train,columns = categorical_columns)
X_test = pd.get_dummies(X_test,columns = categorical_columns)
pipeline=ColumnTransformer([
    ('standard_scaler',StandardScaler(),numeric_columns),
])
pipeline.fit(X_train)
X_train_scaled = pd.concat([pd.DataFrame(pipeline.transform(X_train),index = X_train.index,columns=["age", "bmi","bp","s1","s2","s3","s4","s5","s6"]),X_train[["sex_1","sex_2"]]],axis = 1)
X_test_scaled = pd.concat([pd.DataFrame(pipeline.transform(X_test),index = X_test.index,columns=["age", "bmi","bp","s1","s2","s3","s4","s5","s6"]),X_test[["sex_1","sex_2"]]],axis = 1)
# %%
# SVR
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
svr = SVR()
svr.fit(X_train_scaled, y_train)
y_pred_svr = svr.predict(X_test_scaled)
print("SVR MSE: ")
print(mean_squared_error(y_test, y_pred_svr))
# %%
# Grid Search SVR
param_grid = {'C': [0.1, 1, 10, 100], 
              'gamma': [0.1, 0.01, 0.001,0.0001],
              'kernel': ['rbf']}
grid = GridSearchCV(SVR(), param_grid, cv = 6)
grid.fit(X_train_scaled, y_train)
print("Melhores parametros SVR:")
print(grid.best_params_)
# %%
# SVR Melhores parametros
svr_grid = SVR(C = 100, gamma = 0.01, kernel = 'rbf')
svr_grid.fit(X_train_scaled, y_train)
y_pred_svr_grid = svr_grid.predict(X_test_scaled)
print("MSE SVR Melhores Parametros: ")
print(mean_squared_error(y_test, y_pred_svr_grid))
# %%
# Linear Regression
from sklearn.linear_model import LinearRegression
linear = LinearRegression()
linear.fit(X_train_scaled,y_train)
y_pred_linear = linear.predict(X_test_scaled)
print("MSE Linear Regression: ")
print(mean_squared_error(y_test, y_pred_linear))
# %%
# Random Forest
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
rfr = RandomForestRegressor()
rfr.fit(X_train,y_train)
y_pred_rfr = rfr.predict(X_test)
print("MSE Random Forest: ")
print(mean_squared_error(y_test, y_pred_rfr))
# %%
# Randomized Search Random Forest
n_estimators = [int(x) for x in np.linspace(100,1000,10)]
max_features = [ 'sqrt']
max_depth = [int(x) for x in np.linspace(10,100,10)]
min_samples_split = [2, 4, 5]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split}
rfr_random = RandomizedSearchCV(estimator = RandomForestRegressor(random_state=42), param_distributions = random_grid, cv = 6,random_state=42)
rfr_random.fit(X_train,y_train)
print("Random Forest Melhores Parametros: ")
print(rfr_random.best_params_)
# %%
rfr_random = RandomForestRegressor(n_estimators = 700, min_samples_split=5,max_features='sqrt',max_depth=90,random_state=42)
rfr_random.fit(X_train,y_train)
y_rfr_random = rfr_random.predict(X_test)
print("MSE Random Forest Best Params: ")
print(mean_squared_error(y_test, y_rfr_random))
# %%
# KNN
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor()
knn.fit(X_train_scaled,y_train)
y_pred_knn = knn.predict(X_test_scaled)
print("MSE KNN: ")
print(mean_squared_error(y_test, y_pred_knn))
# %%
# Grid Search KNN
param_grid = {'n_neighbors': range(1,31)}
grid = GridSearchCV(KNeighborsRegressor(), param_grid, cv = 6)
grid.fit(X_train_scaled,y_train)
print("KNN Best Params ")
print(grid.best_params_)
# %%
# KNN Melhores Parametros
knn_best = KNeighborsRegressor(n_neighbors = 18)
knn_best.fit(X_train_scaled,y_train)
y_pred_knn_best = knn_best.predict(X_test_scaled)
print("KNN Best Params")
print(mean_squared_error(y_test, y_pred_knn_best))
# %%
