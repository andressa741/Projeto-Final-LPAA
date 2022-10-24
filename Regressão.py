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
