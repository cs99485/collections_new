from __future__ import print_function
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer as Dv
from sklearn.linear_model import LinearRegression, LassoLars
from sklearn.feature_selection import RFECV
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import ensemble
from matplotlib import pyplot as plt
from sklearn.svm import SVR

train = pd.read_csv("ccfulldata_ntnul.csv")
test = pd.read_csv("ccfulldata_nul.csv")

feature_cols = ['gbs_amount_paid', 'days_to_pay', 'risk_class', 'gbs_payment_term', 'item_family_name',
                'item_product_type', 'ship_type', 'order_type_description', 'route_description', 'business_unit',
                'biczincotrm', 'trx_type', 'amnt_categ' , 'sales_rep_name','product_category','invc_rjctn_desc','reason_coding','customer','payment_days' ]

train = train[feature_cols]
test = test[feature_cols]

print(train.shape)
print(test.shape)

vectorizer = Dv(sparse=False)

train_dict = train.T.to_dict().values()
test_dict = test.T.to_dict().values()
vec_x_cat_train = vectorizer.fit_transform(train_dict) 
vec_x_cat_test = vectorizer.fit_transform(test_dict)
x_train = vec_x_cat_train
x_test = vec_x_cat_test
x_train[np.isnan(x_train)]=0
x_test[np.isnan(x_test)]=0
y_train = train.payment_days
from sklearn.linear_model import LinearRegression 
lm = LinearRegression()
lm.fit(x_train, y_train)
y_test = lm.predict(x_test)
print(y_test)

