#%%
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

import pandas as pd
import numpy as np
from plotnine import *
#%%
dat = pd.read_csv('SalesBook_2013.csv')
 # select variables we will use in class.
 # drop homes that are not single family or duplexes
dat_ml = (dat
     .filter(['NBHD', 'PARCEL', 'LIVEAREA', 'FINBSMNT',  
         'BASEMENT', 'YRBUILT', 'CONDITION', 'QUALITY',
         'TOTUNITS', 'STORIES', 'GARTYPE', 'NOCARS',
         'NUMBDRM', 'NUMBATHS', 'ARCSTYLE', 'SPRICE',
         'DEDUCT', 'NETPRICE', 'TASP', 'SMONTH',
         'SYEAR', 'QUALIFIED', 'STATUS'])
     .rename(columns=str.lower)
     .query('totunits <= 2'))
# %%
# each row represents an individual realestate sale
dat_ml = (dat_ml
    .query('yrbuilt != 0 & condition != "None"')
    .assign(
    before1980 = lambda x: np.where(x.yrbuilt<1980,1,0)
))
# %%
#look at our variables/features and imagine which ones might 
#be good predictors of the age of a house.
(ggplot(dat_ml, aes(x='numbaths.astype(str)', y='yrbuilt')) + 
    geom_boxplot())
# %%
(ggplot(dat_ml, aes(x='quality.astype(str)', y='yrbuilt')) + 
    geom_boxplot())
# https://hcad.org/hcad-resources/hcad-appraisal-codes/hcad-building-grade-adjustment/
# E-, E, E+, D-, D, D+, C-, C, C+, B-, B, B+, A-, A, A+, X-, X, X+

dat_ml.quality.value_counts()
replace_dictionary = {
    "E-":-0.3, "E":0, "E+":0.3,
    "D-":0.7, "D":1, "D+":1.3, 
    "C-":1.7, "C":2, "C+":2.3, 
    "B-":2.7, "B":3, "B+":3.3,
    "A-":3.7, "A":4, "A+":4.3,
    "X-":4.7, "X":5, "X+":5.3
}
qual_ord = dat_ml.quality.replace(replace_dictionary)

# %%
(ggplot(dat_ml, aes(x='arcstyle.astype(str)', y='yrbuilt')) + 
    geom_boxplot() + coord_flip())
# %%
dat_ml.condition.value_counts()
replace_dictionary = {
    "Excel":3,
    "VGood": 2,
    "Good":1,
    "Avg":0,
    "AVG":0,
    "Fair":-1,
    "Poor":-2
}
cond_ord = dat_ml.condition.replace(replace_dictionary)
# %%
# one-hot-encode or dummy variables
# arcstyle, neighborhood(nbhd), gartype

dat_ml.gartype.value_counts()

pd.get_dummies(dat_ml.filter(['arcstyle']), drop_first = True)
# %%
