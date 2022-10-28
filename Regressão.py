# %%
# Import as bibliotecas 

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

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
# Plotando Gráfico
fig, ax = plt.subplots()
ax.scatter(y_pred_svr_grid, y_test)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
plt.title("SVR Best Params")
plt.show()
# %%
# Linear Regression
from sklearn.linear_model import LinearRegression
linear = LinearRegression()
linear.fit(X_train_scaled,y_train)
y_pred_linear = linear.predict(X_test_scaled)
print("MSE Linear Regression: ")
print(mean_squared_error(y_test, y_pred_linear))
fig, ax = plt.subplots()
ax.scatter(y_pred_linear, y_test)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
plt.title("Linear")
plt.show()
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
# Plot do gráfico
fig, ax = plt.subplots()
ax.scatter(y_rfr_random, y_test)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
plt.title("Random Forest Best Params")
plt.show()
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
# Plot do gráfico
fig, ax = plt.subplots()
ax.scatter(y_pred_knn_best, y_test)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
plt.title("KNN Best Params")
plt.show()
# %%
from pickle import dump
dump(svr_grid,open("svr.pkl", 'wb'))
dump(linear,open("lr.pkl", 'wb'))
dump(rfr_random,open("rfr.pkl", 'wb'))
dump(knn_best,open("knnr.pkl", 'wb'))

# %%
# Feature Selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
fs = SelectKBest(score_func=f_regression, k='all')
fs.fit(X_train, y_train)
X_train_fs = fs.transform(X_train)
X_test_fs = fs.transform(X_test)
# Plot dos scores das features
plt.bar([i for i in range(len(fs.scores_))], fs.scores_)
plt.show()
# Conclusão: feature age,s1 e s2 são as menos importantes

pipeline2=ColumnTransformer([
    ('standard_scaler',StandardScaler(),["bmi","bp","s3","s4","s5","s6"]),
])
X_train_removido = X_train.drop(["age","s1","s2"],axis=1)
X_test_removido = X_test.drop(["age","s1","s2"],axis=1)
pipeline2.fit(X_train_removido)

X_train_removido_scaled = pd.concat([pd.DataFrame(pipeline2.transform(X_train_removido),index = X_train_removido.index,columns=[ "bmi","bp","s3","s4","s5","s6"]),X_train_removido[["sex_1","sex_2"]]],axis = 1)
X_test_removido_scaled = pd.concat([pd.DataFrame(pipeline2.transform(X_test_removido),index = X_test_removido.index,columns=[ "bmi","bp","s3","s4","s5","s6"]),X_test_removido[["sex_1","sex_2"]]],axis = 1)
# Treinando algoritmos com X_train_removido_scaled
# SVR
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1, 1, 10, 100], 
              'gamma': [0.1, 0.01, 0.001,0.0001],
              'kernel': ['rbf']}
grid_removido_svr = GridSearchCV(SVR(), param_grid, cv = 6)
grid_removido_svr.fit(X_train_removido_scaled, y_train)
print("Melhores parametros do SVR com feature selection:")
print(grid_removido_svr.best_params_)
svr_removido = SVR(C=100,gamma=0.01,kernel='rbf')
svr_removido.fit(X_train_removido_scaled,y_train)
y_pred_svr_removido = svr_removido.predict(X_test_removido_scaled)
print("MSE SVR com feature selection:")
print(mean_squared_error(y_test,y_pred_svr_removido))
# SVR plot grafico
fig, ax = plt.subplots()
ax.scatter(y_pred_svr_removido, y_test)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
plt.title("SVR Feature Selection")
plt.show()
# KNN
param_grid = {'n_neighbors': range(1,31)}
grid_knn_removido = GridSearchCV(KNeighborsRegressor(), param_grid, cv = 6)
grid_knn_removido.fit(X_train_removido_scaled, y_train)
print("Melhores parametros KNN com feature selection:")
print(grid_knn_removido.best_params_)
knn_removido = KNeighborsRegressor(n_neighbors = 26)
knn_removido.fit(X_train_removido_scaled,y_train)
y_pred_knn_removido = knn_removido.predict(X_test_removido_scaled)
print("MSE KNN com feature selection:")
print(mean_squared_error(y_test,y_pred_knn_removido))
# KNN plot grafico
fig, ax = plt.subplots()
ax.scatter(y_pred_knn_removido, y_test)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
plt.title("KNN Feature Selection")
plt.show()
# Random Forest
n_estimators = [int(x) for x in np.linspace(100,1000,10)]
max_features = [ 'sqrt']
max_depth = [int(x) for x in np.linspace(10,100,10)]
min_samples_split = [2, 4, 5]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split}
rfr_random_removido = RandomizedSearchCV(estimator = RandomForestRegressor(random_state=42), param_distributions = random_grid, n_iter = 40, cv = 6,random_state=42)
rfr_random_removido.fit(X_train_removido,y_train)
print("Melhores parametros Random Forest com Feature Selection:")
print(rfr_random_removido.best_params_)
rfr_removido = RandomForestRegressor(n_estimators = 900, min_samples_split=5,max_features='sqrt',max_depth=50)
rfr_removido.fit(X_train_removido,y_train)
y_pred_rfr_removido = rfr_removido.predict(X_test_removido)
print("MSE Random Forest com feature selection")
print(mean_squared_error(y_test, y_pred_rfr_removido))
# Random Forest plot grafico
fig, ax = plt.subplots()
ax.scatter(y_pred_rfr_removido, y_test)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
plt.title("Random Forest Feature Selection")
plt.show()
# PCA
from sklearn.decomposition import PCA
pca = PCA()
X_train_pca = pca.fit_transform(X_train_scaled)
# Scree Plot
values_pca = np.arange(pca.n_components_)+1
plt.plot(values_pca, pca.explained_variance_ratio_, 'o-', linewidth=2, color='black')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.show()
train_pc1_coord = X_train_pca[:,0]
param_grid = {'C': [0.1, 1, 10, 100], 
              'gamma': [0.1, 0.01, 0.001,0.0001],
              'kernel': ['rbf']}
grid = GridSearchCV(SVR(), param_grid, cv = 6)
grid.fit(train_pc1_coord.reshape(-1,1),y_train)
print("PCA SVR melhores parametros")
print(grid.best_params_)
svr_pca = SVR(C=10,gamma=0.1,kernel='rbf')
svr_pca.fit(train_pc1_coord.reshape(-1,1),y_train)
X_eixo = np.arange(train_pc1_coord.min()-1,train_pc1_coord.max()+1,step=0.1)
y_eixo = svr_pca.predict(X_eixo.reshape(-1,1))
plt.figure()
plt.plot(X_eixo,y_eixo,'k')
plt.scatter(train_pc1_coord,y_train)
# KNN PCA
param_grid = {'n_neighbors': range(1,31)}
grid = GridSearchCV(KNeighborsRegressor(), param_grid, cv = 6)
grid.fit(train_pc1_coord.reshape(-1,1),y_train)
print("Melhores Parametros KNN PCA")
print(grid.best_params_)
knn_pca = KNeighborsRegressor(n_neighbors = 24)
knn_pca.fit(train_pc1_coord.reshape(-1,1),y_train)
y_eixo = knn_pca.predict(X_eixo.reshape(-1,1))
plt.figure()
plt.plot(X_eixo,y_eixo,'k')
plt.scatter(train_pc1_coord,y_train)

# Random Forest
n_estimators = [int(x) for x in np.linspace(100,1000,10)]
max_features = [ 'sqrt']
max_depth = [int(x) for x in np.linspace(10,100,10)]
min_samples_split = [2, 4, 5]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split}
rfr_random = RandomizedSearchCV(estimator = RandomForestRegressor(random_state=42), param_distributions = random_grid, n_iter = 40, cv = 6,random_state=42)
rfr_random.fit(train_pc1_coord.reshape(-1,1),y_train)
print("Melhores parametros Random Forest PCA")
print(rfr_random.best_params_)
rfr_pca = RandomForestRegressor(n_estimators=500,min_samples_split=5,max_features='sqrt',max_depth=10)
rfr_pca.fit(train_pc1_coord.reshape(-1,1),y_train)
y_eixo = rfr_pca.predict(X_eixo.reshape(-1,1))
plt.figure()
plt.plot(X_eixo,y_eixo,'k')
plt.scatter(train_pc1_coord,y_train)
