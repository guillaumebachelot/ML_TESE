#Librairies
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Modules de métriques
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score, auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import plot_confusion_matrix

#Gestion de données déséquilibrées + sensibilité et spécificité
from imblearn.metrics import sensitivity_score
from imblearn.metrics import specificity_score
from imblearn.over_sampling import SMOTE

#Prétraitement (imputation, scaling) et composition de pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

#Selection de variables, de modèles
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score

#Modèles à implémenter
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier

#Divers
import warnings
warnings.filterwarnings("ignore")

#Import datasets
full = pd.read_csv('train_data.csv', sep=';', index_col='Patient') 
print(full.shape)
externe = pd.read_csv("tst_data.csv", sep=';', index_col='Patient') 
print(externe.shape)


#Spliting X and Y
X_full = full.drop(['Classe'], axis=1) 
y_full = full['Classe']
X_ext = externe.drop(['Classe'], axis=1) 
y_ext = externe['Classe']

#preprocessing 
processor =Pipeline([
        ('knn', KNNImputer()), ('standardscaler', StandardScaler())])
processor.fit(X_full)

X_full_2 = processor.fit_transform(X_full)
X_ext_2 = processor.transform(X_ext)

X_full_3 = pd.DataFrame(data = X_full_2, 
                  index = X_full.index, 
                  columns = X_full.columns)
X_ext_3 = pd.DataFrame(data = X_ext_2, 
                  index = X_ext.index, 
                  columns = X_ext.columns)

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
    print('--------------------------------------------------------------------------------------')
    print('--------------------------------------------------------------------------------------')
    print(confusion_matrix(y_ext, prediction2)) 
    print(classification_report(y_ext, prediction2, target_names=['Négative', 'Positive']))
    print("L'accuracy : ", accuracy_score(y_ext, prediction2))
    print("L'AUC ROC pour 1 est de : ", roc_auc_score(y_ext, model.predict_proba(X_ext_3)[:, 1]))
    print('Sensibilité est de :', sensitivity_score(y_ext, prediction2))
    print('Spécificité est de :', specificity_score(y_ext, prediction2))
    plot_confusion_matrix(model, X_ext_3, y_ext)  
    plt.show() 
    sensitivity_score(y_ext, prediction2)
    svc_disp = plot_roc_curve(model, X_ext_3, y_ext)
    plt.show()    


#Internal LOO CV
# Instanciation d'une validation croisée LOO
cv = LeaveOneOut()
# enumerate splits
y_true, y_pred, y_prob = list(), list(), list()
for train_ix, test_ix in cv.split(X_train_3_values):
    # split data
    X_train1, X_test1 = X_train_3_values[train_ix, :], X_train_3_values[test_ix, :]
    y_train1, y_test1 = y_train.iloc[train_ix], y_train.iloc[test_ix]
    # fit model
    model = KNeighborsClassifier()
    model.fit(X_train1, y_train1)
    # evaluate model
    yhat = model.predict(X_test1)
    yprob = model.predict_proba(X_test1)[:, 1]
    # store
    y_true.append(y_test1.iloc[0])
    y_pred.append(yhat[0])
    y_prob.append(yprob[0])
# calculate accuracy
acc = accuracy_score(y_true, y_pred)
print('Accuracy: %.3f' % acc)
from imblearn.metrics import sensitivity_score
from imblearn.metrics import specificity_score
print(model)
print(classification_report(y_true, y_pred))
print('Sensibilité est de : %.3f' % sensitivity_score(y_true, y_pred), '\n')
print('Spécifité est de : %.3f' % specificity_score(y_true, y_pred), '\n')
print(roc_auc_score(y_true, y_prob))

--> FAIRE POUR CHAQUE MODELE 

#Randomizedsearch
skf = StratifiedKFold(n_splits=5, random_state=1234, shuffle=True)

from sklearn.model_selection import RandomizedSearchCV

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


#Show permuration importance
from sklearn.inspection import permutation_importance
mod.fit(X_train_3, y_train)
result = permutation_importance(mod, X_full_3, y_full, 
                                scoring=None, n_repeats=10, n_jobs=None, random_state=1234)

sorted_idx = result.importances_mean.argsort()

fig, ax = plt.subplots(figsize=(12,8))
ax.boxplot(result.importances[sorted_idx].T,
           vert=False, labels=full.columns[sorted_idx])
ax.set_title("Permutation Importances ")
fig.tight_layout()
plt.show()

#RFECV
from sklearn.feature_selection import RFECV
skf = StratifiedKFold(n_splits=5, random_state=1234, shuffle=True)
classifier = RandomForestClassifier(random_state=1234)
min_features_to_select = 1
rfecv = RFECV(estimator=classifier, step=1, cv=skf,
              scoring='accuracy',
              min_features_to_select=min_features_to_select, n_jobs=-1)
rfecv.fit(X_train_3, y_train)
print("Optimal number of features : %d" % rfecv.n_features_)
plt.figure(figsize=(12,8))
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(min_features_to_select,
               len(rfecv.grid_scores_) + min_features_to_select),
         rfecv.grid_scores_)
plt.show()

#Deep learning
from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train),
                                                 y_train)
dict(enumerate(class_weights))

from keras.models import Sequential
from keras.layers import Dense, Dropout

y_dl = tf.keras.utils.to_categorical(y_full, num_classes=2)
y_test_dl = tf.keras.utils.to_categorical(y_test, num_classes=2)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(64, input_shape=(17,), activation='tanh'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(32, activation='tanh'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(2, activation='softmax'))
# binary_crossentropy ou categorical_crossentropy
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=[tf.keras.metrics.Recall(class_id=1), 
                       tf.keras.metrics.AUC(), 
                       tf.keras.metrics.Precision(class_id=1)])

    # Fit data to model
history = model.fit(X_train_3, y_train_dl,
          epochs=200, validation_data=(X_test_3, y_test_dl), shuffle=True,
                    class_weight=dict(enumerate(class_weights)))

# Generate generalization metrics
#cores = model.evaluate(X_full, y, verbose=0)
history_dict=history.history
loss_values = history_dict['loss']
val_loss_values=history_dict['val_loss']
plt.plot(loss_values,'bo',label='training_loss')
plt.plot(val_loss_values,'r',label='val loss')
plt.legend(loc = 'upper right')

a, z, e, r = model.evaluate(X_test_3, y_test_dl)
print('_________________________________')
print('Loss is :', a)
print('Recall is :', z)
print('AUC is :', e)
print('précision is :', r)

