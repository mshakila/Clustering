# CLUSTERING ANALYSIS ASSIGNMENT - crime dataset

# Business Problem: To identify the number of clusters formed and draw inferences.

crime_data <- read.csv('E:/Clustering/crime_data.csv')
# 50 records and  5 variables

''' 
Data Description: The data gives details of crimes with urban population
Murder -- Murder rates in different places of United States
Assault- Assault rate in different places of United States
UrbanPop - urban population in different places of United States
Rape - Rape rate in different places of United States '''

names(crime_data)
# "X"        "Murder"   "Assault"  "UrbanPop" "Rape" 
head(crime_data)

# standardizing or normalizing data: since the values have different range, 
# higher range-values will have more inlfluence
standardized_data <- scale(crime_data[,-1])
head(standardized_data)
mean(standardized_data) # 0
sd(standardized_data) # 0.99 by standardizing mean=0 and std=1

# finding distances between records: distance matrix
dist_eucl <- dist(standardized_data, method='euclidean')
dist_eucl[1:5]

############### Model building
fit_complete <- hclust(dist_eucl, method='complete')
str(fit_complete)
fit_complete$height
fit_complete$order

# display dendrogram
plot(fit_complete, hang=-1)
# we are able to find 2 to 4 clusters

rect.hclust(fit_complete,k=4,border='red')
# k values 2 and 3 are producing uneven clusters. 4 clusters looks good 


###################### ward method
# ward, single , average, centroid
fit_ward <- hclust(dist_eucl, method='ward.D2')
plot(fit_ward, hang=-1)
rect.hclust(fit_ward,k=2,border='red') # tried 2 to 4
# clusters are similar to complete linkage method, 4 clusters

####################### single linkage
fit_single <- hclust(dist_eucl, method='single')
plot(fit_single, hang=-1)
rect.hclust(fit_single,k=4,border='red') # tried 2 to 4
# here clusters are overlapping, cannot obtain good clusters

####################### centroid linkage
fit_centroid <- hclust(dist_eucl, method='centroid')
plot(fit_centroid, hang=-1)
rect.hclust(fit_centroid,k=2,border='red') # tried 2 to 4
# in centroid clusters are overlapping, cannot obtain good clusters

####################### average linkage
fit_average <- hclust(dist_eucl, method='average')
plot(fit_average, hang=-1)
rect.hclust(fit_average,k=2,border='red') # tried 2 to 4
# in average, its hard to differentiate btw clusters, but can obtain 2 clusters

# we have used euclidean distance and tried different clustering methods.
# We have obtained 4 clusters through hierarchial clustering 

################### lets try kmeans
set.seed(100) 
fit_kmeans <- kmeans(standardized_data,4,nstart=10)
str(fit_kmeans)
fit_kmeans$cluster
fit_kmeans$centers

library(animation)
fit_kmeans <- kmeans.ani(standardized_data,4)
# here used only 2 variables

# lets use principal components to plot the clusters
library(cluster)
clusplot(standardized_data, fit_kmeans$cluster, color = T, labels = 2, main = 'Cluster Plot')
# the 4 clusters look good and 87% variablity is explained by the 2 components.


####################### final model using complete linkage method
fit_complete <- hclust(dist_eucl, method='complete')

groups <- cutree(fit_complete,k=4)
membership <- as.matrix(groups)
table(membership)
#  1  2  3  4 
#  8 11 21 10

final <- data.frame(crime_data,membership)
head(final)

library(data.table) # to change order of columns and make membership as col1
setcolorder(final,c('membership'))

# finding averages of all clusters, to draw insights
aggregate(crime_data[,-1],by=list(final$membership),mean)

'''
From the 4 clusters formed, we can distinguish them based on crime rate intensity 

Groups        Attributes                   Names
  1      highest murder rates          Unsafe cities
  2      high in all crimes            Highly unsafe cities
  3      second lowest crime rates     Second Safest cities
  4      lowest crime rate             Safest cities



CONCLUSIONS

The business problem is to cluster the cities based on crime rates. Here there 
is no dependent variable. We have to use unsupervised learning algorithms. Since the
data has to be grouped, we are using clustering algorithm.

We have built models using euclidean distance method and various linkages. The 
data wass first standardized to remove the influence of high range values. The
different methods used are ward, single-linakge, complete-linkage, average-linkage,
centroid and kmeans.

We were able to separate the cities into 4 main groups. One of the groups named
safest cities had the least crime rate and lowest urban population.

The clustering method has helped to group the cities: so that the law
enforcement agencies know where more vigilance and control is needed.

'''