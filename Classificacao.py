# %%
# Import as bibliotecas 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %%
# Import do dataset
from sklearn.datasets import load_breast_cancer
X,y = load_breast_cancer(return_X_y= True, as_frame = True)
# %%
# Tratamento e preparação dos dados
print(load_breast_cancer().DESCR)
# %%
X.head()
# %%
y.head()
# %%
X.info()
# %%
X.isnull().sum()
# %%
X.describe()
# %%
plt.figure()
sns.boxplot(x="mean radius", data = X)
plt.show()
plt.figure()
sns.boxplot(x="mean texture", data = X)
plt.show()
plt.figure()
sns.boxplot(x="mean perimeter", data = X)
plt.show()
plt.figure()
sns.boxplot(x="mean area", data = X)
plt.show()
# %%
correlacao = X.corr()
abs_correlacao = abs(correlacao)
sns.set(rc = {'figure.figsize':(18,18)})
plt.figure()
sns.heatmap(correlacao, annot = True,cmap="Reds")
plt.show()
# %%
upper_abs_correlacao= abs_correlacao.where(np.triu(np.ones(correlacao.shape),k=1).astype(bool))
upper_abs_correlacao = upper_abs_correlacao.unstack().dropna()
upper_abs_correlacao[upper_abs_correlacao>0.95]
# %%
X_removed = X.drop(['mean radius','mean area', 'radius error'],axis = 1)
# %%
