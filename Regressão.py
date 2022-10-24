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
