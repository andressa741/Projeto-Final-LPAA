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
# Algoritmos de aprendizado
# SVM
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

# %%
clf_svm = SVC()
clf_svm.fit(X_train_scaled_standard,y_train)
y_pred = clf_svm.predict(X_test_scaled_standard)
print(classification_report(y_test,y_pred))
# %%
param_grid = {'C': [0.1, 1, 10, 100], 
              'gamma': [0.1, 0.01, 0.001],
              'kernel': ['rbf']}
# %%
grid = GridSearchCV(SVC(), param_grid, cv = 4)
grid.fit(X_train_scaled_standard, y_train)
print(grid.best_params_)
# %%
clf_svm = SVC(C = 10, gamma = 0.001, kernel = 'rbf')
clf_svm.fit(X_train_scaled_standard, y_train)
y_pred = clf_svm.predict(X_test_scaled_standard)
print(classification_report(y_test,y_pred))
# %%
# Confusion matrix 
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure()
sns.heatmap(cm, cmap = "Greens", annot=True, 
            cbar_kws = {"orientation":"vertical","label":"color bar"},
            xticklabels = [0,1], yticklabels = [0,1]);
plt.xlabel('Predicted labels');plt.ylabel('True labels');plt.title("Confusion Matrix: SVM") 
plt.show()
# %%