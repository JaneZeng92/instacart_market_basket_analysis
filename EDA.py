######## python library #########
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from sklearn.model_selection import train_test_split


from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()



######## ignore warnings #########

import warnings
warnings.filterwarnings("ignore")

# import dataset and clean the data

# https://www.kaggle.com/c/instacart-market-basket-analysis

orders = pd.read_csv("orders.csv")
print(orders.head(5))
orders.info()

departments = pd.read_csv("departments.csv")
print(departments.head(5))
departments.info()

prior = pd.read_csv("order_products__prior.csv")
print(prior.head(5))
prior.info()

products = pd.read_csv("products.csv")
print(products.head(5))
products.info()

train = pd.read_csv("order_products__train.csv")
print(train.head(5))
train.info()

aisles = pd.read_csv("aisles.csv")
print(aisles.head(5))
aisles.info()


###################################################################

### Plot

plt.figure()
sns.countplot(x="order_hour_of_day", data=orders, color= 'skyblue')
plt.xlabel('Hours of Day')
plt.ylabel("Counts")
plt.title('Frequency of Order Hours of Day')
plt.show()

plt.figure()
sns.countplot(x="order_dow", data=orders, color= 'red')
plt.xlabel('Day of Week')
plt.ylabel("Counts")
plt.title('Frequency of Order by Week Day')
plt.show()

plt.figure()
sns.countplot(x="days_since_prior_order", data=orders, color= 'navy')
plt.xlabel('Number of Days')
plt.ylabel("Counts")
plt.xticks(rotation='vertical')
plt.title('Number of Days Since Prior Order')
plt.show()

grouped_orders = orders.groupby(["order_dow", "order_hour_of_day"])["order_number"].aggregate("count").reset_index()
grouped_orders = grouped_orders.pivot('order_dow', 'order_hour_of_day', 'order_number')
plt.figure()
sns.heatmap(grouped_orders)
plt.xlabel('Hours of Day')
plt.ylabel("Day of Week")
plt.title("Frequency of Day of week Vs Hour of day")
plt.show()


########################################################

# percentage of re-orders in prior set #
reorder_prior = prior.reordered.sum() / prior.shape[0]
print(reorder_prior)

grouped_train = train.groupby("order_id")["add_to_cart_order"].aggregate("max").reset_index()
grouped_count = grouped_train.add_to_cart_order.value_counts()

plt.figure(figsize=(12,8))
sns.barplot(grouped_count.index, grouped_count.values, alpha=0.8)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Number of Products in the Each Order', fontsize=12)
plt.title('Number of Products in Each Order')
plt.xticks(rotation='vertical')
plt.show()

# products frequency counts
merge = pd.merge(prior, products, on='product_id')
merge = pd.merge(merge, aisles, on='aisle_id')
merge = pd.merge(merge, departments, on='department_id')
merge = pd.merge(merge, orders, on='order_id')
pd.set_option('max_columns', None)
print(merge.head(5))

merge_aisle = merge['aisle'].value_counts().head(20)
plt.figure(figsize=(12, 8))
sns.barplot(merge_aisle.index, merge_aisle.values, alpha=0.8)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Aisle', fontsize=12)
plt.title('Aisle Count')
plt.xticks(rotation='vertical')
plt.show()


merge_department = merge['department'].value_counts().head(20)
plt.figure(figsize=(12, 8))
sns.barplot(merge_department.index, merge_department.values, alpha=0.8)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Departments', fontsize=12)
plt.title('Departments Distribution')
plt.xticks(rotation='vertical')
plt.show()

