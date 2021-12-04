#%%
#import sys
#!{sys.executable} -m pip install scikit-learn dalex shap
# %%
import pandas as pd
import numpy as np
import joblib # to savel ml models
from plotnine import *

from sklearn import metrics

from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
#%%
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
# %%
dat_ml = pd.read_pickle('dat_ml.pkl')
# %%
# Split our data into training and testing sets.
X_pred = dat_ml.drop(['yrbuilt', 'before1980'], axis = 1)
y_pred = dat_ml.before1980
X_train, X_test, y_train, y_test = train_test_split(
    X_pred, y_pred, test_size = .34, random_state = 76)  
#%%








#%%
logreg = LogisticRegression(max_iter = 100000, 
                            solver = 'newton-cg', 
                            penalty = 'l2')
logreg.fit(X_train,y_train)
test_log = logreg.predict_proba(X_test)
auc = metrics.roc_auc_score(y_test, test_log[:,1])
y_pred = logreg.predict(X_test)
#%%
print('Train Accuracy:', logreg.score(X_train, y_train))
print('Test Accuracy:', logreg.score(X_test, y_test))
print("Recall Score:", metrics.recall_score(y_test,y_pred))
print("Logistic Regression AUC:", auc)
#%%
#liblinear and l1 got 85% and 84% accuracies
#newton-cg and l2 similar (slightly better)
#%%










#%%
mlpc = MLPClassifier(max_iter = 10000,
                    hidden_layer_sizes = (10,9,8,7,6,5,4,3,2,1),
                    activation = 'identity',
                    solver = 'adam',
                    batch_size = 10,
                    learning_rate = 'adaptive')

mlpc.fit(X_train, y_train)
#%%
test_logmp = mlpc.predict_proba(X_test)
#aucmlp = metrics.roc_auc_score(y_test, test_logmp[:,1])
y_predmlp = mlpc.predict(X_test)

#%%
print("Train Accuracy:", mlpc.score(X_train, y_train))
print("Test Accuracy:", mlpc.score(X_test, y_test))
#print("Multilayer Perceptron AUC:", aucmlp)
print("Recall Score:", metrics.recall_score(y_test,y_predmlp))
#%%









#%%
from sklearn.ensemble import RandomForestClassifier
#%%
forest = RandomForestClassifier(criterion = 'entropy',
                                n_estimators = 20,
                                max_depth = 20,
                                min_samples_split = 5,
                                min_samples_leaf = 2,
                                max_features = 'auto')

forest.fit(X_train, y_train)

test_logf = forest.predict_proba(X_test)
aucf = metrics.roc_auc_score(y_test, test_logf[:,1])
y_predf = forest.predict(X_test)
#%%
print("Train Accuracy:", forest.score(X_train,y_train))
print("Test Accuracy:", forest.score(X_test, y_test))
print("Random Forest AUC:", aucf)
print("Recall Score:", metrics.recall_score(y_test,y_predf))
#%%







#%%
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
#%%
dtc = tree.DecisionTreeClassifier(criterion = 'entropy',
                                  splitter = 'best',
                                  max_depth = 14,
                                  min_samples_split = 3,
                                  min_samples_leaf = 6)
dtc = dtc.fit(X_train, y_train)

test_logdt = dtc.predict_proba(X_test)
aucdt = metrics.roc_auc_score(y_test, test_logdt[:,1])
y_preddt = dtc.predict(X_test)
#%%
print("Train Accuracy:", dtc.score(X_train,y_train))
print("Test Accuracy:",dtc.score(X_test,y_test))
print("Decision Tree Classifier AUC:", aucdt)
print("Recall Score:", metrics.recall_score(y_test,y_preddt))
#%%










#%%
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#%%
xgb = XGBClassifier(booster = 'dart')
xgb.fit(X_train, y_train)

test_logx = xgb.predict_proba(X_test)
aucx = metrics.roc_auc_score(y_test, test_logx[:,1])

#%%
print("Train Accuracy:", xgb.score(X_train, y_train))
print("Test Accuracy:",xgb.score(X_test,y_test))
print("XGBoost AUC:", aucx)
#%%











#%%
# build models
clfNB = GaussianNB()
clfGB = GradientBoostingClassifier()
clfNB.fit(X_train, y_train)
clfGB.fit(X_train, y_train)
ypred_clfNB = clfNB.predict(X_test)
ypred_clfGB = clfGB.predict(X_test)
# %%
ypred_clfGB_prob = clfGB.predict_proba(X_test)
# %%
metrics.plot_roc_curve(clfGB, X_test, y_test)
metrics.plot_roc_curve(clfNB, X_test, y_test)
# %%
metrics.confusion_matrix(y_test, ypred_clfNB)
#%%
metrics.confusion_matrix(y_test, ypred_clfGB)
# %%
df_features = pd.DataFrame(
    {'f_names': X_train.columns, 
    'f_values': clfGB.feature_importances_}).sort_values('f_values', ascending = False).head(12)

# Python sequence slice addresses 
# can be written as a[start:end:step]
# and any of start, stop or end can be dropped.
# a[::-1] is reverse sequence.
f_names_cat = pd.Categorical(
    df_features.f_names,
    categories=df_features.f_names[::-1])

df_features = df_features.assign(f_cat = f_names_cat)

(ggplot(df_features,
    aes(x = 'f_cat', y = 'f_values')) +
    geom_col() +
    coord_flip() +
    theme_bw()
    )
#%%
# Variable Reduction
# build reduced model
compVars = df_features.f_names[::-1].tolist()

X_pred_reduced = dat_ml.filter(compVars, axis = 1)
y_pred = dat_ml.before1980

X_train_reduced, X_test_reduced, y_train, y_test = train_test_split(
    X_pred_reduced, y_pred, test_size = .34, random_state = 76)  
#%%
clfGB_reduced = GradientBoostingClassifier()
clfGB_reduced.fit(X_train_reduced, y_train)
ypred_clfGB_red = clfGB_reduced.predict(X_test_reduced)

# %%
print(metrics.classification_report(ypred_clfGB_red, y_test))
metrics.confusion_matrix(y_test, ypred_clfGB_red)
# %%
# saving models as python objects 
joblib.dump(clfNB, 'models/clfNB.pkl')
joblib.dump(clfGB, 'models/clfGB.pkl')
joblib.dump(clfGB_reduced, 'models/clfGB_final.pkl')
df_features.f_names[::-1].to_pickle('models/compVars.pkl')
# %%
