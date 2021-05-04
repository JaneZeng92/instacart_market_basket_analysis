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


orders = pd.read_csv("orders.csv")
departments = pd.read_csv("departments.csv")
prior = pd.read_csv("order_products__prior.csv")
products = pd.read_csv("products.csv")
train = pd.read_csv("order_products__train.csv")
aisles = pd.read_csv("aisles.csv")


merge = pd.merge(prior, products, on='product_id')
merge = pd.merge(merge, aisles, on='aisle_id')
merge = pd.merge(merge, departments, on='department_id')
merge = pd.merge(merge, orders, on='order_id')
pd.set_option('max_columns', None)
print(merge.head(5))

################ Clustering Customers

product_count = merge['product_name'].value_counts().reset_index().head(10)
product_count.columns = ['product_name', 'frequency_count']
print(product_count)

aisle_count = merge['aisle'].value_counts().reset_index().head(10)
aisle_count.columns = ['aisle', 'frequency_count']
print(aisle_count)

print(len(merge['product_name'].unique()))
print(len(merge['aisle'].unique()))

customer_prod = pd.crosstab(merge['user_id'], merge['aisle'])
# print(customer_prod.head(10))

clusters = range (1, 10)
value  = []
for k in clusters:
    model = KMeans(n_clusters = k)
    model.fit(customer_prod)
    value.append(model.inertia_)

plt.figure()
plt.plot(clusters, value)
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.xticks(clusters)
plt.show()


pca = PCA(n_components=6).fit(customer_prod)
pca = PCA().fit(customer_prod)
pca_userandorder = pca.transform(customer_prod)
print(pca.explained_variance_ratio_.sum())
components = pd.DataFrame(pca_userandorder)
print(components.head())

kmean = KMeans(n_clusters = 6)
cluster = kmean.fit_predict(pca_userandorder)

color_dict = {0:'blue', 1:'purple', 2:'yellow', 3:'black', 4:'red', 5:'green'}
color = [color_dict[i] for i in cluster]
# label_dict = {0:'1', 1:'2', 2:'3', 3:'4', 4:'5', 5:'6'}
# label = [label_dict[j] for j in cluster]
plt.figure(figsize = (15,8))
plt.scatter(pca_userandorder[:,0],pca_userandorder[:,2], c=color, alpha=0.3)
plt.xlabel('X-Values')
plt.ylabel('Y-Values')
plt.title('K-mean Plot with 6 Clusters')
# plt.legend()
plt.show()

customer_prod['cluster'] = cluster
sort = customer_prod['cluster'].value_counts().sort_values(ascending=False)
print(sort)

cluster0 = customer_prod[customer_prod['cluster']==0].drop('cluster',axis=1).mean()
print(cluster0.sort_values(ascending=False)[0:10])
cluster1 = customer_prod[customer_prod['cluster']==1].drop('cluster',axis=1).mean()
print(cluster1.sort_values(ascending=False)[0:10])
cluster2 = customer_prod[customer_prod['cluster']==2].drop('cluster',axis=1).mean()
print(cluster2.sort_values(ascending=False)[0:10])
cluster3 = customer_prod[customer_prod['cluster']==3].drop('cluster',axis=1).mean()
print(cluster3.sort_values(ascending=False)[0:10])
cluster4 = customer_prod[customer_prod['cluster']==4].drop('cluster',axis=1).mean()
print(cluster4.sort_values(ascending=False)[0:10])
cluster5 = customer_prod[customer_prod['cluster']==5].drop('cluster',axis=1).mean()
print(cluster5.sort_values(ascending=False)[0:10])
