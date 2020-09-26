###### ASSIGNMENT CLUSTERING crime dataset

# Business Problem: To cluster the dataset into similar groups based on crime rates

# Data set collection and details
# The dataset gives details of crimes committed in various US cities. 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# laoding the datasset
crime = pd.read_csv('E:/Clustering/crime_data.csv')
crime.columns
# 'Unnamed: 0', 'Murder', 'Assault', 'UrbanPop', 'Rape'

crime = crime.rename(columns ={'Unnamed: 0':'Cities'}) # renaming first col
# 'Cities', 'Murder', 'Assault', 'UrbanPop', 'Rape'

# Normalizing the numeric variables: to bring all to same scale
def norm_func (i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)
a=pd.Series([1,2,3,4,5]) # checking the func
norm_func(a)

crime_norm = norm_func(crime.iloc[:,1:])

######## Dendrogram to find number of clusters
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

model_comp = linkage(crime_norm, method='complete', metric='euclidean')

%matplotlib qt
plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(model_comp,leaf_font_size=8. ,leaf_rotation=0.)
''' dendrogram illustrates the process of cluster formation. From the figure we can see
that the first cluster was formed between indices 14 and 28. The last two clusters
were formed at a distance of around 1.6
From the dendrogram it is clear that we can form 2 to 4 good clusters
'''
help(linkage)
# using linkage methods are single, complete, average, weighted and ward, displaying dendrograms
model1 = linkage(crime_norm,method='single')
sch.dendrogram(model1)
model1 = linkage(crime_norm,method='average')
sch.dendrogram(model1)
model1 = linkage(crime_norm,method='weighted')
sch.dendrogram(model1)
model1 = linkage(crime_norm,method='ward')
sch.dendrogram(model1)
''' 
for single linkage - overlapping clusters
for average - no clear clusters
for weighted - 2 clusters, if choose 4: random sized clusters (one is very small)
for ward - 2 or 4 clusters '''

############## Model building 

from sklearn.cluster import AgglomerativeClustering
h_complete = AgglomerativeClustering(n_clusters=3, linkage='complete',affinity='euclidean').fit(crime_norm)

cluster_labels =pd.Series(h_complete.labels_)

crime['clust_complete'] = cluster_labels
crime.head()

# getting aggregate mean of each cluster
crime.iloc[:,1:5].groupby(crime.clust_complete).mean()

''' 
From the 3 clusters formed, we can distinguish them based on crime rate intensity 

Groups        Attributes                   Names
  0      medium crime rate          Unsafe cities
  1      lowest crime rate          Safest cities 
  2      highest in all crimes      Highly unsafe cities
'''

########## using ward method
model_ward = linkage(crime_norm, method='ward')
sch.dendrogram(model_ward)
h_ward	=	AgglomerativeClustering(n_clusters=2, linkage='ward',affinity = "euclidean").fit(crime_norm) 
cluster_labels_ward=pd.Series(h_ward.labels_)
crime['clust_ward']=cluster_labels_ward
crime.head()
crime.iloc[:,1:5].groupby(crime.clust_ward).mean()

crime = crime.iloc[:,[5,6,0,1,2,3,4]]
crime.head()

''' 
From the 2 clusters formed, we can distinguish them based on crime rate intensity 

Groups        Attributes                   Names
  0      lowest crime rate          Safest cities 
  1      highest in all crimes      Highly unsafe cities
'''
# creating ans saving results in a csv file 
crime.to_csv("E:/Clustering/crime_clust.csv")

'''
CONCLUSIONS

The business problem is to cluster the cities based on crime rates. Here there 
is no dependent variable. We have to use unsupervised learning algorithms. Since the
data has to be grouped, we are using clustering algorithm. Here we have used 
Agglomerative hierarchial clustering since the dataset is small.

We have built models using euclidean distance method and various linkages. The 
data wass first standardized to remove the influence of high range values. The
different methods used are ward, single-linakge, complete-linkage, average-linkage,
weighted and ward.

When used complete linkage method, we were able to separate the cities into 3 main 
groups: safe, unsafe and highly unsafe. By using ward's method, we could separate 
cities into 2 main groups: safe and unsafe cities

Clustering method can help the law enforcement agencies to increase their vigilance 
in unsafe cities. 

'''





