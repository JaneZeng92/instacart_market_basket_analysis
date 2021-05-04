import numpy as np
import pandas as pd
import gc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import xgboost as xgb

####################

import warnings
warnings.filterwarnings("ignore")

####################
orders = pd.read_csv("orders.csv")
departments = pd.read_csv("departments.csv")
prior = pd.read_csv("order_products__prior.csv")
products = pd.read_csv("products.csv")
train = pd.read_csv("order_products__train.csv")
aisles = pd.read_csv("aisles.csv")

# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
#
#
# param_grid = {
#     'eavl_metric' : 'logloss',
#     'max_depth' : 6,
#     'colsample_bytree' : 0.4,
#     'subsample' : 0.8}
#
#
# xgb_clf = xgb.XGBClassifier(objective='binary:logistic', parameters=param_grid, num_boost_round=10)
#
# model = xgb_clf.fit(X_train, y_train)
#
# xgb.plot_importance(model)
# plt.show()
#
# y_pred = xgb_clf.predict(X_test).astype('int')

################### Feature Prediction
merge = pd.merge(prior, products, on='product_id')
merge = pd.merge(merge, aisles, on='aisle_id')
merge = pd.merge(merge, departments, on='department_id')
merge = pd.merge(merge, orders, on='order_id')
pd.set_option('max_columns', None)
print(merge.head(5))

total_order = merge.groupby('user_id')['order_number'].max().to_frame('total_orders')
total_order = total_order.reset_index()
user_reorder = merge.groupby('user_id')['reordered'].mean().to_frame('user_reorder_ratio')
user_reorder = user_reorder.reset_index()
# print(total_order.head(5))
# print(user_reorder.head(5))
total_user_reorder = pd.merge(total_order, user_reorder, on='user_id')
gc.collect()
print(total_user_reorder.head(5))

total_ppurchases = merge.groupby('product_id')['order_id'].count().to_frame('total_ppurchases')
total_ppurchases = total_ppurchases.reset_index()
total_ppurchases = total_ppurchases.loc[total_ppurchases['total_ppurchases'] > 100]

product_reorder = merge.groupby('product_id')['reordered'].mean().to_frame('product_reorder_ratio')
product_reorder = product_reorder.reset_index()
# print(total_ppurchases.head(5))
# print(product_reorder.head(5))
total_product_reorder = pd.merge(total_ppurchases, product_reorder, on='product_id')
gc.collect()
print(total_product_reorder.head(5))

user_product_order = merge.groupby(['user_id','product_id'])['order_id'].count().to_frame('user_total_product')
user_product_order = user_product_order.reset_index()

purchase_time = merge.groupby(['user_id','product_id'])['order_id'].count().to_frame('purchase_time')
# purchase_time = purchase_time.reset_index()
print(purchase_time.head(5))

first_time_order = merge.groupby(['user_id','product_id'])['order_number'].min().to_frame('first_time_order')
first_time_order = first_time_order.reset_index()
print(first_time_order.head(5))

combine = pd.merge(total_order, first_time_order, on = 'user_id',how = 'right')
# print(combine.head(5))
combine['combine'] = combine.total_orders-combine.first_time_order + 1
print(combine.head(5))

purchase_ratio = pd.merge(purchase_time, combine, on = ['user_id', 'product_id'])
# print(purchase_ratio.head(5))
purchase_ratio['purchase_ratio'] = purchase_ratio['purchase_time']/purchase_ratio['combine']
# print(purchase_ratio.head(5))
upo_pr= pd.merge(user_product_order, purchase_ratio, on = ['user_id', 'product_id'])
# print(upo_pr.head(5))

df_new = pd.merge(upo_pr, total_user_reorder, on ='user_id')
df_new = pd.merge(df_new , total_product_reorder, on = 'product_id')
df_new = df_new[['user_id', 'product_id', 'user_reorder_ratio', 'product_reorder_ratio', 'purchase_ratio']]
print(df_new.head(5))

order = orders[['user_id', 'order_id']]
order_pred = df_new.merge(order, on = 'user_id', how = 'left')
print(order_pred.head(5))

########################### Feature Important

merge2 = pd.merge(orders, train, on = 'order_id')
merge2 = pd.merge(merge2, products, on = 'product_id' )
pd.set_option('max_columns', None)
print(merge2.head(5))

x = merge2[['order_id', 'user_id', 'order_number', 'product_id']]
y = merge2.reordered
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

param_grid = {
    'eavl_metric' : 'logloss',
    'max_depth' : 5,
    'colsample_bytree' : 0.4,
    'subsample' : 0.8}


xgb_clf = xgb.XGBClassifier(objective='binary:logistic', parameters=param_grid, num_boost_round=10)
model_xgb = xgb_clf.fit(X_train, y_train)
xgb.plot_importance(model_xgb)
plt.show()

param = {
    'max_depth' : [4,5,6,7],
    'colsample_bytree' : [0.2,0.3,0.4,0.5],
}

grid = GridSearchCV(xgb_clf,param, cv=3, verbose=2, n_jobs=1)
model_grid = grid.fit(X_train, y_train)
print(model_grid.best_score_)
print(model_grid.best_params_)

pred = (model_grid.predict_proba(y_test)[:,1]>= 0.2)
merge2['prediction'] = pred
merge2.head(5)