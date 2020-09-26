# CLUSTERING ANALYSIS ASSIGNMENT - EastWestAirlines dataset

# Business Problem: To identify the number of clusters formed and draw inferences.

library(readxl)
airlines <- data.frame(read_excel('E:/Clustering/EastWestAirlines.xlsx',sheet = 2))
names(airlines)[names(airlines)=='Award.'] <- "Award"  # renaming last column
# 3999 records and  12 variables

names(airlines)

# "ID#"               "Balance"           "Qual_miles"        "cc1_miles"        
# "cc2_miles"         "cc3_miles"         "Bonus_miles"       "Bonus_trans"      
# "Flight_miles_12mo" "Flight_trans_12"   "Days_since_enroll" "Award"   

summary(airlines)
# all variables are considered as numeric type

# Also,for  cc_miles following bins have been created
# 1 = under 5,000
# 2 = 5,000 - 10,000
# 3 = 10,001 - 25,000
# 4 = 25,001 - 50,000
# 5 = over 50,000  

str(airlines) # showing all as numeric data
attach(airlines)


#### normalizing  the variables
normalize <- function(x){
  return((x-min(x))/(max(x)-min(x)))
}
a <- c(1,2,3,4,5)  
normalize(a)
normalized_air <- normalize(airlines[,-1])

#### finding distance between records
dist <- dist(normalized_air, method='euclidean')

################# Model building
######### Hierarchial cluster
fit_complete <- hclust(dist, method='complete')
str(fit_complete)

# display dendrogram
plot(fit_complete, hang=-1)
rect.hclust(fit_complete,k=2,border='red')
# since there are many  obs the dendrogram cannot be properly visualized,
# also cannot separate clusters

fit_ward <- hclust(dist, method='ward.D2')
plot(fit_ward, hang=-1)
rect.hclust(fit_ward,k=2,border='red')
# using ward's method we are able to identify 2 clusters (highly unequal size)

groups <- cutree(fit_ward,k=2)
membership <- as.matrix(groups)
table(membership)

# fit_ward and k=4
# 1    2    3    4 
# 3215  134  624   26 

# fit_ward and k=2
#  1     2  
#  3215  784

# fit_complete and k=2
# 1    2 
# 3989   10 

# fit_complete and k=4
# 1    2    3    4 
# 3963   26    9    1 

# let us use fit_ward method with k=2 clusters
final_hclust <- data.frame(airlines,membership)

library(data.table) # to change order of columns and make membership as col1
setcolorder(final_hclust,c('membership'))

# finding averages of all clusters, to draw insights
aggregate(airlines[,-c(1,2)],by=list(final_hclust$membership),mean)
prop.table(table(final_hclust$membership))

# we can group the data into 2 clusters. Group-1 has 80% of the customers and
# 20% in Group-2. Group-2 fliers having higher values in all the variables. 
# We can say that Group_2 is more loyal and have had more free-flights (awards).
# The marketing team can target Group-1 to increase their miles (and/or transactions)

################## K-MEANS clustering

########## first let us find out how many clusters have to be selected
######### k = sqrt(n/2)
sqrt(3999/2) # 44 clusters
# if we use 44 clusters it beats the aim of clustering 

########## k selection
library(kselection)
library(doParallel)
registerDoParallel(cores=2)
k <- kselection(normalized_air,parallel = TRUE, k_threshold = 0.90, max_centers = 44)
k # finds 2 clusters , 
# k_threshold values chosen were 0.85,0.9,0.95,1.00 - for all got 2 clusters

####### elbow or scree plot
set.seed(123)
twss <- NULL
for (i in 1:12)
  twss[i] = sum(kmeans(normalized_air, centers=i)$tot.withinss)

windows()
plot(1:12, twss, type = 'b',)

### or use fviz_nbclust function to get same results
library(factoextra)
set.seed(123)
fviz_nbclust(normalized_air,kmeans, method = 'wss') # showing 2 clusters
fviz_nbclust(normalized_air,kmeans, method = 'silhouette') # showing 2 clusters

######### Running k-means algorithm

set.seed(123)
kmeans2 <- kmeans(normalized_air,2) # using 2 clusters 
# as suggested by elbow plot , kselection and average-silhouette methods
str(kmeans2)
kmeans2$centers
# for 2 clusters size        : int [1:2] 330 3669
# to reach to final result done 1 iteration, with totss 14.8 and total-withinss
# 7.16 (sum of withinss 4.21 and 2.96)
# 
############ visualize the results
fviz_cluster(kmeans2, data = normalized_air)
# plots first 2 prinicpal components (rather than 2 variables like kmeans.ani)
km <- kmeans.ani(normalized_air,2)
# shows 2 clear clusters using variables Qual_miles and Balance

############ using k=3
kmeans3 <- kmeans(normalized_air,3)
str(kmeans3)
# 3 clusters of size 3118,81,800 
fviz_cluster(kmeans3,normalized_air)

########## using clara function 
xcl <- clara(normalized_air, 2, sample = 500)
clusplot(xcl)
fviz_cluster(xcl, data = normalized_air)

############### Partitioning around medoids
xpm <- pam(normalized_air, 2)
clusplot(xpm)

########### using kmeans with 3 clusters
final_kmeans3 <- data.frame(airlines,kmeans3$cluster)
setcolorder(final_kmeans3,neworder = 'kmeans3.cluster')
head(final_kmeans3)
aggregate(airlines[,-1], by=list(kmeans3$cluster), FUN=mean)

# cluster-2 has highest values followed by cluster-3 for most of the variables.
# cluster-1 has the least values. 

################ let us use kmeans with 2 clusters as final 
final_kmeans2<- data.frame(airlines, kmeans2$cluster) # append cluster membership

setcolorder(final_kmeans2, neworder = c("kmeans2.cluster"))
head(final_kmeans2)
aggregate(airlines[,-1], by=list(kmeans2$cluster), FUN=mean)
# cluster-1 has high values for all (except cc2_miles and cc3_miles)

'''
CONCLUSIONS

We have to separate the dataset into clusters. Since the data has no dependant
variable, we are using unsupervised learning technique. Since we have to group
the data, we are using clustering algorithms. 

We have used different distance measures. we have used euclidean distance, 
Gower distance metrics. 

To know optimum number of clusters, we used k-selection, elbow plot
we have used different clustering methods. The methods used are hierarchial 
clustering (using complete linkage, Ward method ), Kmeans, clara and 
PAM

We were able to form 2 clusters that were easily distinguishable. Gower
distance matrix clubbed with hierarchial-clustering gave us best results

'''