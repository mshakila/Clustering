# CLUSTERING ANALYSIS ASSIGNMENT - EastWestAirlines dataset 
# using Gower distance metric for mixed data types

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

str(airlines)

unique(airlines$cc1_miles) #  1 4 3 2 5
table(airlines$cc1_miles)
# 1    2    3    4    5 
# 2289  284  613  525  288 

unique(airlines$cc2_miles) # 1 2 3
table(airlines$cc2_miles)
# 1    2    3 
# 3956   28   15 

unique(airlines$cc3_miles) # 1 3 2 4 5
table(airlines$cc3_miles)
# 1    2    3    4    5 
# 3981    3    4    6    5 

unique(airlines$Qual_miles) # 0 and many more entries
table(airlines$Qual_miles)
# 0 - 3773 obs

unique(airlines$Bonus_miles) # 0 and many more entries
table(airlines$Bonus_miles)
# 0 - 475 obs

unique(airlines$Bonus_trans) # 0 and 57 more entries
table(airlines$Bonus_trans)
# 0 - 475 obs

unique(airlines$Flight_miles_12mo) # 0 and many more entries
table(airlines$Flight_miles_12mo)
# 0 - 2723 obs

unique(airlines$Flight_trans_12) # 0 and 35 more entries
table(airlines$Flight_trans_12)
# 0 - 2723 obs

unique(airlines$Days_since_enroll) # 0 and many more entries
table(airlines$Days_since_enroll)
# no 0 records

unique(airlines$Award) # 0 and 1
table(airlines$Award)
# 0    1 
# 2518 1481 

attach(airlines)

# feature engineering
airlines_new <-  airlines
# for below variables converting 0=0 and more-than-0 as 1
# Bonus_miles,Bonus_trans,Flight_miles_12mo,Flight_trans_12,Days_since_enroll,`Award?`

airlines_new['Qual_miles_new'] <- ifelse(airlines_new$Qual_miles==0,0,1)
airlines_new['Bonus_miles_new'] <- ifelse(airlines_new$Bonus_miles==0,0,1)
airlines_new['Bonus_trans_new'] <- ifelse(airlines_new$Bonus_trans==0,0,1)
airlines_new['Flight_miles_12mo_new'] <- ifelse(airlines_new$Flight_miles_12mo==0,0,1)
airlines_new['Flight_trans_12_new'] <- ifelse(airlines_new$Flight_trans_12==0,0,1)
airlines_new['Days_since_enroll_new'] <- ifelse(airlines_new$Days_since_enroll==0,0,1)

head(airlines_new,2)
str(airlines_new)
# All variables mentioned as numeric, let us convert them to their
# approriate types in airlines_new dataset

# Continuous variables are 7: Balance, Qual_miles, Bonus_miles, Bonus_trans,
# Flight_miles_12mo, F, light_trans_12, Days_since_enroll

# Factor variables are 7: 'Award','Qual_miles_new', 'Bonus_miles_new', 'Bonus_trans_new',
# 'Flight_miles_12mo_new', 'Flight_trans_12_new', 'Days_since_enroll_new'

# Ordinal variables are 3: 'cc1_miles', 'cc2_miles', 'cc3_miles', 
# 
# ID variable can be safely assumed as character variable

######## converting to factor variables
air <- airlines_new
cols_factor <- c('Award','Qual_miles_new', 'Bonus_miles_new', 'Bonus_trans_new',
                 'Flight_miles_12mo_new', 'Flight_trans_12_new', 
                 'Days_since_enroll_new')
air[,cols_factor] <- data.frame(apply(air[cols_factor],2,as.factor))

########## converting to ordinal variables
air$cc1_miles <- factor(air$cc1_miles,order=TRUE)
levels(air$cc1_miles)

air$cc2_miles <- factor(air$cc2_miles,order=TRUE)
levels(air$cc2_miles)

air$cc3_miles <- factor(air$cc3_miles,order=TRUE)
levels(air$cc3_miles)

str(air)

################# Model building
######### Hierarchial cluster

# since our dataset has mixed data types , we will consider Gower distance metric
# r function to calculate Gower dist is daisy
# Here, for each daya type a particular distance metric that suits that type is
# calculated. Then linear combination with weights is used to claculate final
# distance metrix.
# continuous data: manhattan distance
# nominal data: by tweaking manhattan distance
# ordinal data: dice coefficient

gower_dist <- daisy(air[,-1], metric = 'gower')
summary(gower_dist)
# Types = I, I, O, O, O, I, I, I, I, I, N, N, N, N, N, N, N 
# the metric has correctly taken data types I-interval, o-ordinal, n-nominal

gower_mat <- as.matrix(gower_dist)

# choosing number of clusters
k <- kselection(gower_dist,parallel = TRUE, k_threshold = 0.90, max_centers = 44)
k

fviz_nbclust(gower_dist,kmeans, method = 'wss') # showing 2 clusters
fviz_nbclust(gower_dist,kmeans, method = 'silhouette') # showing 2 clusters

fit_gow_complete <- hclust(gower_dist, method='complete')
str(fit_gow_complete)

# display dendrogram
plot(fit_gow_complete, hang=-1)
rect.hclust(fit_gow_complete,k=2,border='red')

groups <- cutree(fit_gow_complete,k=2)
membership <- as.matrix(groups)
table(membership)
#   1    2 
# 2723 1276 

final_gower <- data.frame(air,membership)

setcolorder(final_gower,'membership')
str(final_gower)
head(final_gower,2)
library(modeest) # to find mode -mfv (most frequent value)

aggregate(final_gower[,c(3,4,8,9,10,11,12)],by=list(final_gower$membership),mean) # calculating mean for interval variables
aggregate(final_gower[,-c(1,2,3,4,8,9,10,11,12)], by=list(final_gower1$membership),mfv) # finding mode for nominal and ordinal variables

# Let us find the characteristics of cluster-1 fliers
# They have low eligible miles for Award(Balance), low Qual_miles
# they have earned less bonus_miles from non_flight transactions, have less non-flight transactions
# they have been in the fleir program for lesser number of days
# They have not got any free-flights(Award is 0), also for the past
# 12-months they have no transactions or miles (both are 0)

# using gower distance matrix we are able to diffentiate the clusters better than
# using normal distance matrices. 
