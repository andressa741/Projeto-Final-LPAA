# %%
# Import as bibliotecas 
from random import Random
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
from sklearn.model_selection import StratifiedKFold
param_grid = {'C': [0.1, 1, 10, 100], 
              'gamma': [0.1, 0.01, 0.001],
              'kernel': ['rbf']}
# %%
grid = GridSearchCV(SVC(), param_grid, cv = StratifiedKFold(
        n_splits=4,
        shuffle=True,
        random_state=42
    ))
grid.fit(X_train_scaled_standard, y_train)
print(grid.best_params_)
# %%
clf_svm = SVC(C = 10, gamma = 0.001, kernel = 'rbf',probability=True)
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
# Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train,y_train)
feature_importances = pd.Series(rf.feature_importances_, index=X.columns)
feature_importances.sort_values(inplace=True)
feature_importances.plot(kind='barh')
# %% 
from sklearn.feature_selection import SelectKBest, f_classif
bestfeatures = SelectKBest(score_func=f_classif, k=11)
best_features = bestfeatures.fit(X_train,y_train)
dfscores = pd.Series(best_features.scores_,index=X.columns)
dfscores.sort_values(inplace=True)
dfscores.plot(kind='barh')
# %%
# Random Forest
rfc = RandomForestClassifier(n_estimators=50,random_state=42)
rfc.fit(X_train,y_train)
y_pred_base_model = rfc.predict(X_test)
print(classification_report(y_test,y_pred_base_model))
print(rfc.get_params())
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_base_model)
plt.figure()
sns.heatmap(cm, cmap = "Greens", annot=True, 
            cbar_kws = {"orientation":"vertical","label":"color bar"},
            xticklabels = [0,1], yticklabels = [0,1]);
plt.xlabel('Predicted labels');plt.ylabel('True labels');plt.title("Confusion Matrix: Random Forest") 
plt.show()
# %%
# Randomized Search Random Forest
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
n_estimators = [int(x) for x in np.linspace(100,1000,10)]
max_features = [ 'sqrt']
max_depth = [int(x) for x in np.linspace(10,100,10)]
min_samples_split = [2, 4, 5]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split}
rf_random = RandomizedSearchCV(estimator = RandomForestClassifier(random_state=42), param_distributions = random_grid, n_iter = 40, cv = StratifiedKFold(
        n_splits=4,
        shuffle=True,
        random_state=42
    ), scoring = 'accuracy',random_state=42)
rf_random.fit(X_train,y_train)
print("Melhores Parametros Random Forest:")
print(rf_random.best_params_)
# %%
# Random Forest com melhores parametros
rfc_random = RandomForestClassifier(n_estimators = 100, min_samples_split=2,max_features='sqrt',max_depth=40,random_state=42)
rfc_random.fit(X_train,y_train)
y_pred_random = rfc_random.predict(X_test)
print(classification_report(y_test,y_pred_random))
# %%
# Confusion Matrix Random Forest Melhores Parametros
cm = confusion_matrix(y_test, y_pred_random)
plt.figure()
sns.heatmap(cm, cmap = "Greens", annot=True, 
            cbar_kws = {"orientation":"vertical","label":"color bar"},
            xticklabels = [0,1], yticklabels = [0,1]);
plt.xlabel('Predicted labels');plt.ylabel('True labels');plt.title("Confusion Matrix: Random Forest Best Params") 
plt.show()
# %%
# KNN
from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train_scaled_standard, y_train)
y_pred_knn = knn_clf.predict(X_test_scaled_standard)
print(classification_report(y_test,y_pred_knn))
# %%
# Confusion Matrix KNN 
cm = confusion_matrix(y_test, y_pred_knn)
plt.figure()
sns.heatmap(cm, cmap = "Greens", annot=True, 
            cbar_kws = {"orientation":"vertical","label":"color bar"},
            xticklabels = [0,1], yticklabels = [0,1]);
plt.xlabel('Predicted labels');plt.ylabel('True labels');plt.title("Confusion Matrix: KNN") 
plt.show()
# %%
# Grid Search KNN
param_grid = {'n_neighbors': range(1, 31)}
grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid, cv=StratifiedKFold(
        n_splits=4,
        shuffle=True,
        random_state=42
    ))
grid_knn.fit(X_train_scaled_standard, y_train)
print("KNN Melhores Parametros")
print(grid_knn.best_params_)
# %%
# KNN Melhores Parametros
knn_grid = KNeighborsClassifier(n_neighbors=8)
knn_grid.fit(X_train_scaled_standard,y_train)
y_pred_grid_knn = knn_grid.predict(X_test_scaled_standard)
print(classification_report(y_test,y_pred_grid_knn))
# %%
# KNN Confusion Matrix Melhores parametros
cm = confusion_matrix(y_test, y_pred_grid_knn)
plt.figure()
sns.heatmap(cm, cmap = "Greens", annot=True, 
            cbar_kws = {"orientation":"vertical","label":"color bar"},
            xticklabels = [0,1], yticklabels = [0,1]);
plt.xlabel('Predicted labels');plt.ylabel('True labels');plt.title("Confusion Matrix: KNN Best Params") 
plt.show()
# %%
from pickle import dump
dump(clf_svm,open("svm.pkl", 'wb'))
dump(rfc_random,open("rfc.pkl", 'wb'))
dump(knn_grid,open("knn.pkl", 'wb'))



# %%
# Plot Curva ROC
pred_prob_svm = clf_svm.predict_proba(X_test_scaled_standard)
pred_prob_rfc = rfc_random.predict_proba(X_test)
pred_prob_knn = knn_grid.predict_proba(X_test_scaled_standard)
from sklearn.metrics import roc_curve
fpr_svm, tpr_svm, thresh_svm = roc_curve(y_test, pred_prob_svm[:,1])
fpr_rfc, tpr_rfc, thresh_rfc = roc_curve(y_test, pred_prob_rfc[:,1])
fpr_knn, tpr_knn, thresh_knn = roc_curve(y_test, pred_prob_knn[:,1])
plt.plot(fpr_svm, tpr_svm, linestyle='--',color='orange', label='SVM')
plt.plot(fpr_rfc, tpr_rfc, linestyle='--',color='blue', label='Random Forest')
plt.plot(fpr_knn, tpr_knn, linestyle='--', color='green',label='KNN')

plt.title('ROC curve')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive rate')

plt.legend(loc='best')
plt.show()