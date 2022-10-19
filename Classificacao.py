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
# Separação dos dados em treinamento e teste
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
X_removed_train, X_removed_test, y_train, y_test = train_test_split(X_removed, y, test_size = 0.2, stratify = y, random_state = 42)
# %%
# Feature Scaling
from sklearn.preprocessing import StandardScaler
scaling_standard = StandardScaler()
scaling_standard.fit(X_train)
X_train_scaled_standard = scaling_standard.transform(X_train)
X_test_scaled_standard = scaling_standard.transform(X_test)
scaling_removed_standard = StandardScaler()
scaling_removed_standard.fit(X_removed_train)
X_removed_train_scaled_standard = scaling_removed_standard.transform(X_removed_train)
X_removed_test_scaled_standard = scaling_removed_standard.transform(X_removed_test)
# %%
