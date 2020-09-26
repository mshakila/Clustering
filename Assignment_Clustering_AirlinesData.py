# ASSIGNMENT CLUSTERING using EastwestAirlines dataset

# Business Problem: To identify different clusters/ groups of fliers 

# Data set collection and details
# The dataset gives information on passengers who belong to an airline’s frequent flier program

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

airlines = pd.read_excel('E:/Clustering/EastWestAirlines.xlsx','data')
airlines.columns
airlines = airlines.rename(columns ={'Award?': 'Award'}) # renaming last col

airlines.describe()
airlines.info() # all cols datatypes given as interval

# finding missing values
airlines.isna().sum() # no missing values

train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)

airlines[['Balance','Award']].groupby(['Balance'], as_index=False).mean().sort_values(by='Award', ascending=False)

# let us normalize the data 
def norm_func(i):
    x = (i - i.min()) / (i.max() - i.min())
    return x

air_norm = norm_func(airlines.iloc[:,1:]) # removing ID col
air_norm.head(3)

######## let us find number of clusters using Dendrogram 
import scipy.cluster.hierarchy as sch

model_comp = sch.linkage(air_norm, method='complete', metric='euclidean')

%matplotlib qt
plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(model_comp,leaf_font_size=8. ,leaf_rotation=0.)
# clearly showing 2 clusters (if choose 4 clusters, size variation is very high)

############## Model building 

from sklearn.cluster import AgglomerativeClustering
h_complete2 = AgglomerativeClustering(n_clusters=2, linkage='complete', affinity ='euclidean').fit(air_norm)

cluster_labels =pd.Series(h_complete2.labels_)
air = airlines.copy()

air['clust_complete2'] = cluster_labels
air.head()

# getting aggregate mean of each cluster
air_aggregate=air.iloc[:,1:12].groupby(air.clust_complete2).mean()

'''
Interpreting the clusters
cluster-1 has low values in all except cc2_miles, cc3_miles and Days_since_enroll
cluster-2 has higher values in all except the above mentioned 3 variables.
Let us now look at K-means clustering algorithm 
'''
### 3 clusters
h_complete3 = AgglomerativeClustering(n_clusters=3, linkage='complete', affinity ='euclidean').fit(air_norm)
cluster_labels =pd.Series(h_complete3.labels_)
air['clust_complete3'] = cluster_labels
air_aggregate=air.iloc[:,1:12].groupby(air.clust_complete3).mean()

### 4 clusters
h_complete4 = AgglomerativeClustering(n_clusters=4, linkage='complete', affinity ='euclidean').fit(air_norm)
cluster_labels =pd.Series(h_complete4.labels_)
air['clust_complete4'] = cluster_labels
air_aggregate=air.iloc[:,1:12].groupby(air.clust_complete4).mean()

# using 3 and 4 clusters cannot properly differentiate the clusters

################### K-Means clustering

## finding optimal number of clusters
# from dendrogram and running Agglomerative algorithm, we have found 2 clusters to be optimal

# finding optimal clusters using elbow plot
k=list(range(2, 11))
k
TWSS = []
for i in k:
    kmeans = KMeans(n_clusters=i).fit(air_norm)
    WSS = []
    for j in range(i):
        WSS.append(sum(cdist(air_norm.iloc[kmeans.labels_ == j,:], kmeans.cluster_centers_[j].reshape(1,air_norm.shape[1]),'euclidean')))
    TWSS.append(sum(WSS))

TWSS_clusters = pd.DataFrame(columns=['TWSS','No. of clusters'])
TWSS_clusters.TWSS = pd.Series(TWSS)
TWSS_clusters['No. of clusters'] = pd.Series(k)

# Scree plot 
plt.plot(k,TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(k);
# showing clusters as 5 (we can see a slight elbow at 4, but clear elbow at 5)

# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model=KMeans(n_clusters=5, random_state=0) 
model.fit(air_norm)

model.labels_ # getting the labels of clusters assigned to each row 
md=pd.Series(model.labels_)  # converting numpy array into pandas series object 
air['clust_kmeans5']=md # creating a  new column and assigning it to new column 
air.head()

air_aggregate=air.iloc[:,1:12].groupby(air.clust_kmeans5).mean()

# but we have seen that 2 clusters is better
kmeans2 = KMeans(n_clusters=2, random_state=0).fit(air_norm)
md = pd.Series(kmeans2.labels_)
air['clust_kmeans2']=md
air_aggregate=air.iloc[:,1:12].groupby(air.clust_kmeans2).mean()

air.to_csv("E:/Clustering/air_clust.csv")

# cluster-2 has highest values in all variables
# With kmeans we have got a better model.
'''
we have now got 2 segments of our fliers. These segments can now be targeted for different types of mileage offers
cluster-2 are already enjoying offers like free-flights (Awards). 
For cluster-1, mean Balance (miles elgible for award travel) is around 60000 against 97000 of cluster-2
So, for cluster-1 fliers discounts can be given for next few flights if they reach say, 70000 miles.
