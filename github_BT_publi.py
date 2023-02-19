"""
This document shows the codes of the different steps whose results are presented in 
the article "Testicular sperm extraction in non-obstructive azoospermia context: Machine Learning approach".
The code was adapted from a jupyter notebook, which runs cell by cell.

"""
#importing the libraries and packages needed to complete the project

import pandas as pd
import numpy as np
import seaborn as sns
import scipy
import matplotlib.pyplot as plt
import tensorfow as tf

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import plot_confusion_matrix

from imblearn.metrics import sensitivity_score
from imblearn.metrics import specificity_score

from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import LeaveOneOut

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

import warnings
warnings.filterwarnings("ignore")

#Import datasets (training and external test set)
full = pd.read_csv('train_data.csv', sep=';', index_col='Patient') 
externe = pd.read_csv("test_data.csv", sep=';', index_col='Patient') 


#Spliting X (features) and Y (target)
X_full = full.drop(['Classe'], axis=1) 
y_full = full['Classe']
X_ext = externe.drop(['Classe'], axis=1) 
y_ext = externe['Classe']

#Preprocessing 
processor =Pipeline([
        ('knn', KNNImputer()), ('standardscaler', StandardScaler())])

#apply processing
X_full_2 = processor.fit_transform(X_full)
X_ext_2 = processor.transform(X_ext)

#rebuilt DataFrame 
X_full_3 = pd.DataFrame(data = X_full_2, 
                  index = X_full.index, 
                  columns = X_full.columns)
X_ext_3 = pd.DataFrame(data = X_ext_2, 
                  index = X_ext.index, 
                  columns = X_ext.columns)
# Exploratory data analysis (only the code for test set exploration is presented here)
for i in externe.columns.values:
    if externe[i].dtype == np.number:
        print(i)
        sub = externe[i][externe['Classe']==1].dropna().values
        fert = externe[i][externe['Classe']==0].dropna().values
        stat, p = scipy.stats.shapiro(externe[i].dropna().values)
        alpha = 0.05
        if p > alpha:
            stat3, p3 = scipy.stats.bartlett(sub, fert)           
            if p3 > 0.05:
                kk = True
            else:
                kk = False
            print("t-test/welch-test")
            print('Mean (SD) = %0.2f +/- %0.2f' % (sub.mean(), sub.std()))
            print('Mean (SD) = %0.2f +/- %0.2f' % (fert.mean(), fert.std()))
            stat1, p1 = scipy.stats.ttest_ind(sub, fert, equal_var=kk)
            print(p1)    
        else:
            print('mannwitney')
            stat2, p2 = scipy.stats.mannwhitneyu(sub, fert, alternative='two-sided', use_continuity=True)
            print(p2)
            print('Median (IQR) = %0.2f +/- %0.2f' % (np.median(sub), scipy.stats.iqr(sub)))
            print('Median (IQR) = %0.2f +/- %0.2f' % (np.median(fert), scipy.stats.iqr(fert)))
        print("\n")
    else:
        print(i)
        print('chi2 ')
        conting = pd.crosstab(externe[i], externe['Classe'], margins = False)
        chi2, p,  dof, expected = scipy.stats.chi2_contingency(conting)
        print(chi2)
        print(p)
        print("\n")

##training, optimizing, and evaluation

#First screening 
liste_models = [LogisticRegression(solver='liblinear'), 
                RandomForestClassifier(random_state=1234), 
                GradientBoostingClassifier(random_state=1234, subsample=0.8, max_features='auto'), 
                XGBClassifier(random_state=1234), 
                GaussianNB(), 
                SVC(probability=True),
               KNeighborsClassifier()]
for model in liste_models:
    model.fit(X_full_3, y_full)
    prediction2 = model.predict(X_ext_3)
    print(confusion_matrix(y_ext, prediction2)) 
    print(classification_report(y_ext, prediction2, target_names=['negative', 'positive']))
    print("L'accuracy : ", accuracy_score(y_ext, prediction2))
    print("L'AUC ROC pour 1 est de : ", roc_auc_score(y_ext, model.predict_proba(X_ext_3)[:, 1]))
    print('Sensibilité est de :', sensitivity_score(y_ext, prediction2))
    print('Spécificité est de :', specificity_score(y_ext, prediction2))
    plot_confusion_matrix(model, X_ext_3, y_ext)  
    svc_disp = plot_roc_curve(model, X_ext_3, y_ext)
    plt.show()    

#Internal CV (example)
from sklearn.model_selection import RepeatedKFold
from numpy import mean
from numpy import std
from sklearn.neural_network import MLPClassifier
cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
from sklearn.metrics import recall_score, make_scorer
liste_models = [LogisticRegression(solver='liblinear'), 
                GaussianNB(), 
                RandomForestClassifier(random_state=1234), 
                GradientBoostingClassifier(random_state=1234, subsample=0.8, max_features='auto'), 
                XGBClassifier(random_state=1234), 
                SVC(probability=True),
               KNeighborsClassifier()]
for model in liste_models:
    # evaluate model
    scores = cross_val_score(model, X_full_3, y_full, scoring='recall', cv=cv, n_jobs=-1)
    # report performance
    print('--------------------------------------------------------------------------------------')
    print(model)
    print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
    print('--------------------------------------------------------------------------------------')
#optimizing / Randomizedsearch (example)
skf = StratifiedKFold(n_splits=5, random_state=1234, shuffle=True)
gbt_model = GradientBoostingClassifier(random_state=1234, subsample=0.8, max_features='auto')
                     
gbt_params = {
    "loss":["deviance"],
    "learning_rate": [0.01, 0.025, 0.05, 0.1, 0.2],
    "min_samples_split": np.linspace(0.1, 0.5, 3),
    "min_samples_leaf": np.linspace(0.1, 0.5, 3),
    "max_depth":[3,5,8],
    "max_features":["log2","sqrt"],
    "criterion": ["friedman_mse",  "mae"],
    "subsample":[0.5, 0.618, 0.85, 0.95, 1.0],
    "n_estimators":[10],
    'random_state': [1234]}

gbt_cv_model = RandomizedSearchCV(gbt_model, gbt_params, cv=skf, n_jobs=-1).fit(X_full_3, y_full)
print(gbt_cv_model.best_params_)


#Show permuration importance (example)
from sklearn.inspection import permutation_importance
mod.fit(X_full_3, y_full)
result = permutation_importance(mod, X_full_3, y_full, 
                                scoring=None, n_repeats=10, n_jobs=None, random_state=1234)
sorted_idx = result.importances_mean.argsort()
fig, ax = plt.subplots(figsize=(12,8))
ax.boxplot(result.importances[sorted_idx].T,
           vert=False, labels=full.columns[sorted_idx])
ax.set_title("Permutation Importances ")
fig.tight_layout()
plt.show()

# Learning curve plotting (on training set only)
from sklearn.model_selection import learning_curve
N, train_score, val_score = learning_curve(mod, X_full_3, y_full,
                                          cv=5, scoring='roc_auc',
                                           train_sizes=np.linspace(0.1, 1, 10))

plt.figure(figsize=(8, 6))
plt.plot(N, train_score.mean(axis=1), label='train score')
plt.plot(N, val_score.mean(axis=1), label='validation score')
plt.legend()

#Deep learning - neural network example
from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train),
                                                 y_train)
dict(enumerate(class_weights))

y_dl = tf.keras.utils.to_categorical(y_full, num_classes=2)
y_test_dl = tf.keras.utils.to_categorical(y_test, num_classes=2)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(64, input_shape=(16,), activation='tanh'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(32, activation='tanh'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=[tf.keras.metrics.Recall(class_id=1), 
                       tf.keras.metrics.AUC(), 
                       tf.keras.metrics.Precision(class_id=1)])
history = model.fit(X_full_3, y_dl,
          epochs=200, validation_data=(X_test_3, y_test_dl), shuffle=True,
                    class_weight=dict(enumerate(class_weights)))

history_dict=history.history
loss_values = history_dict['loss']
val_loss_values=history_dict['val_loss']
plt.plot(loss_values,'bo',label='training_loss')
plt.plot(val_loss_values,'r',label='val loss')
plt.legend(loc = 'upper right')
