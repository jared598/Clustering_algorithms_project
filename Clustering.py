# Importing the needed libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
import time
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import sklearn

name = ['Q-E', 'ZN-E', 'PH-E', 'DBO-E', 'DQO-E', 'SS-E', 'SSV-E', 'SED-E', 'COND-E', 'PH-P', 'DBO-P', 'SS-P', 'SSV-P',
        'SED-P', 'COND-P',
        'PH-D', 'DBO-D', 'DQO-D', 'SS-D', 'SSV-D', 'SED-D', 'COND-D', 'PH-S', 'DBO-S', 'DQO-S', 'SS-S', 'SSV-S',
        'SED-S', 'COND-S',
        'RD-DBO-P', 'RD-SS-P', 'RD-SED-P', 'RD-DBO-S', 'RD-DQO-S', 'RD-DBO-G', 'RD-DQO-G', 'RD-SS-G', 'RD-SED-G']

# Importing the datasets
dowjones = pd.read_csv(r"C:\Users\GGPC\Desktop\AUT\data mining\dow_jones_index.data")
sales = pd.read_csv(r"C:\Users\GGPC\Desktop\AUT\data mining\Sales_Transactions_Dataset_Weekly.csv")
facebook = pd.read_csv(r"C:\Users\GGPC\Desktop\AUT\data mining\facebook Live_20210128.csv")
treatment = pd.read_csv(r"C:\Users\GGPC\Desktop\AUT\data mining\water-treatment.data", names=name, na_values='?')

# Feature Selection for the Dow Jones dataset


# Checking for missing values in the data set
dowjones.isnull().sum()

# Deleting the missing values since they are just 30 and thus not much
dowjones = dowjones.dropna()

# Removing the $ sign in the open,high,low,close,next_week_open and next_weeks_close
dowjones['open'] = dowjones.open.str.replace("$", " ")
dowjones['high'] = dowjones.high.str.replace("$", " ")
dowjones['low'] = dowjones.low.str.replace("$", " ")
dowjones['close'] = dowjones.close.str.replace("$", " ")
dowjones['next_weeks_open'] = dowjones.next_weeks_open.str.replace("$", " ")
dowjones['next_weeks_close'] = dowjones.next_weeks_close.str.replace("$", " ")

# Converting the object variables that are actually quantitative to float
dowjones['open'] = dowjones['open'].astype(float)
dowjones['high'] = dowjones['high'].astype(float)
dowjones['low'] = dowjones['low'].astype(float)
dowjones['close'] = dowjones['close'].astype(float)
dowjones['next_weeks_open'] = dowjones['next_weeks_open'].astype(float)
dowjones['next_weeks_close'] = dowjones['next_weeks_close'].astype(float)
dowjones['quarter'] = dowjones['quarter'].astype(object)

# Obtaining the y and X variables
y = dowjones['percent_change_next_weeks_price']
X = dowjones.drop(['percent_change_next_weeks_price', 'date'], axis=1)

# Dummy coding the predictors
X = pd.get_dummies(X, drop_first=True)

# Running the ensemble to generate the feature importance values
extra_tree = ExtraTreesRegressor(n_estimators=500, random_state=1)
extra_tree.fit(X, y)

feat_labels = X.columns
importances = extra_tree.feature_importances_
indices = np.argsort(importances)[::-1]

# Plotting the feature importances
plt.figure(figsize=(10, 7))
plt.title('Feature Importance')
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), feat_labels[indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.tight_layout()
plt.show()

# Obtaining the features to use for the clustering (All the explanatory variables except Stock)
X_clustering = dowjones.drop(['percent_change_next_weeks_price', 'date', 'stock', 'quarter', 'low', 'open', 'high'],
                             axis=1)

# Feature Selection for the Sales dataset


# Obtaining the y and X variables
X = sales.drop(['W51', 'MAX', 'MIN', 'Product_Code'], axis=1)

# Standardizing the variables
X = StandardScaler().fit_transform(X)

# Obtaining the principal components
pca = PCA(n_components=6)
principalComponents = pca.fit_transform(X)

# Extracting the 1st principal component to use as feature for the clustering
salesPComp = pd.DataFrame(data=principalComponents[:, 0], columns=['1st principal component'])

# Feature Selection for the Facebook dataset


# Obtaining the y and X variables
X = facebook.drop(['status_published', 'status_id'], axis=1)

# Dummy coding the status_type
X = pd.get_dummies(X, drop_first=True)

# Standardizing the variables
X = StandardScaler().fit_transform(X)

# Obtaining the principal components
pca = PCA(n_components=10)
principalComponents = pca.fit_transform(X)

# Extracting the first three principal component to use as features for the clustering
facebookPComps = pd.DataFrame(data=principalComponents[:, 0:3],
                              columns=['1st principal component', '2nd principal component', '3rd principal component'])

# ### Feature Selection for the water treatment dataset


# Checking for missing values in the data set
treatment.isnull().sum()

# Imputing the missing values using the mean of the variables since deleting the missing values eliminated 147 datapoints
treatment = treatment.fillna(treatment.mean())
X = treatment

# Standardizing the variables
X = StandardScaler().fit_transform(X)

# Obtaining the principal components
pca = PCA(n_components=10)
principalComponents = pca.fit_transform(X)

# Extracting the first three principal component to use as features for the clustering
treatmentPComps = pd.DataFrame(data=principalComponents[:, 0:3],
                               columns=['1st principal component', '2nd principal component',
                                        '3rd principal component'])

# ### Task 1, Part a
# ### Obtaining the K-means clustering on each of the dataset
# ### For the DowJones Dataset


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clustering)

silhouette_coefficients = []
sse = []

t = time.process_time()
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, random_state=200)
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_)
    score = silhouette_score(X_scaled, kmeans.labels_)
    silhouette_coefficients.append(score)
elapsed_time = time.process_time() - t
print("The time elapsed is:", elapsed_time)

plt.figure(figsize=(20, 5))
plt.subplot(1, 2, 1)
plt.title('SSE as a function of no of clusters')
plt.plot(range(2, 11), sse)
plt.xticks(range(2, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")

plt.subplot(1, 2, 2)
plt.plot(range(2, 11), silhouette_coefficients)
plt.title('Silhouette coefficients as a function of no of clusters')
plt.xticks(range(2, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Coefficient")

plt.show()

# For the sales Dataset

scaler = StandardScaler()
X_scaled = scaler.fit_transform(salesPComp)

silhouette_coefficients = []
sse = []

t = time.process_time()
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, random_state=200)
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_)
    score = silhouette_score(X_scaled, kmeans.labels_)
    silhouette_coefficients.append(score)
elapsed_time = time.process_time() - t
print("The time elapsed is:", elapsed_time)

plt.figure(figsize=(20, 5))
plt.subplot(1, 2, 1)
plt.plot(range(2, 11), sse)
plt.title('SSE as a function of no of clusters')
plt.xticks(range(2, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")

plt.subplot(1, 2, 2)
plt.plot(range(2, 11), silhouette_coefficients)
plt.title('Silhouette Coefficients as a function of no of clusters')
plt.xticks(range(2, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Coefficient")

plt.show()

# For the facebook Dataset


scaler = StandardScaler()
X_scaled = scaler.fit_transform(facebookPComps)

silhouette_coefficients = []
sse = []

t = time.process_time()
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, random_state=200)
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_)
    score = silhouette_score(X_scaled, kmeans.labels_)
    silhouette_coefficients.append(score)
elapsed_time = time.process_time() - t
print("The time elapsed is:", elapsed_time)

plt.figure(figsize=(20, 5))
plt.subplot(1, 2, 1)
plt.plot(range(2, 11), sse)
plt.title('SSE as a function of no of clusters')
plt.xticks(range(2, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")

plt.subplot(1, 2, 2)
plt.plot(range(2, 11), silhouette_coefficients)
plt.title('Silhouette Coefficients as a function of no of clusters')
plt.xticks(range(2, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Coefficient")

plt.show()

# For the treatment Dataset


scaler = StandardScaler()
X_scaled = scaler.fit_transform(treatmentPComps)

silhouette_coefficients = []
sse = []

t = time.process_time()
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, random_state=200)
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_)
    score = silhouette_score(X_scaled, kmeans.labels_)
    silhouette_coefficients.append(score)
elapsed_time = time.process_time() - t
print("The time elapsed is:", elapsed_time)

plt.figure(figsize=(20, 5))
plt.subplot(1, 2, 1)
plt.plot(range(2, 11), sse)
plt.xticks(range(2, 11))
plt.title('SSE as a function of no of clusters')
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")

plt.subplot(1, 2, 2)
plt.plot(range(2, 11), silhouette_coefficients)
plt.xticks(range(2, 11))
plt.title('Silhouette coefficient as a function of no of clusters')
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Coefficient")

plt.show()

# Obtaining the 4 by 3 table for the Kmeans Clustering


results = [['DowJones', 6.875, 2672.8410421064514, 0.28648575], ['Sales', 2.75, 9.290948527316536, 0.7702601],
           ['Facebook', 33.984, 4027.666498076413, 0.71788], ['Treatment', 5.59375, 715.086199, 0.292312]]
result = pd.DataFrame(results, columns=['Dataset', 'Time taken', 'Sum of Squares Error', 'Cluster Silhouette Measure'])
print(result)

# Displaying the CSM plot for the best value of the K parameter for each dataset

# For the DowJones Dataset


scaler = StandardScaler()
X = scaler.fit_transform(X_clustering)

range_n_clusters = [5]

for n_clusters in range_n_clusters:
    fig, (ax1) = plt.subplots(1, 1)
    fig.set_size_inches(18, 7)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
    clusterer = KMeans(n_clusters=n_clusters, random_state=200)
    cluster_labels = clusterer.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    sample_silhouette_values = silhouette_samples(X, cluster_labels)
    y_lower = 10

    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10
    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])
    ax1.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.suptitle(("Silhouette analysis for KMeans clustering on the DowJones dataset "
                  "with n_clusters = %d" % n_clusters), fontsize=14, fontweight='bold')
plt.show()

# Displaying the CSM plot for the best value of the K parameter for each dataset

# For the Sales  Dataset


scaler = StandardScaler()
X = scaler.fit_transform(salesPComp)

range_n_clusters = [5]

for n_clusters in range_n_clusters:
    fig, (ax1) = plt.subplots(1, 1)
    fig.set_size_inches(18, 7)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
    clusterer = KMeans(n_clusters=n_clusters, random_state=200)
    cluster_labels = clusterer.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    sample_silhouette_values = silhouette_samples(X, cluster_labels)
    y_lower = 10

    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10
    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])
    ax1.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.suptitle(("Silhouette analysis for KMeans clustering on the Sales dataset "
                  "with n_clusters = %d" % n_clusters), fontsize=14, fontweight='bold')
plt.show()

# Displaying the CSM plot for the best value of the K parameter for each dataset

# For the Facebook  Dataset


scaler = StandardScaler()
X = scaler.fit_transform(facebookPComps)

range_n_clusters = [5]

for n_clusters in range_n_clusters:
    fig, (ax1) = plt.subplots(1, 1)
    fig.set_size_inches(18, 7)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
    clusterer = KMeans(n_clusters=n_clusters, random_state=200)
    cluster_labels = clusterer.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    sample_silhouette_values = silhouette_samples(X, cluster_labels)
    y_lower = 10

    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10
    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])
    ax1.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.suptitle(("Silhouette analysis for KMeans clustering on the facebook dataset "
                  "with n_clusters = %d" % n_clusters), fontsize=14, fontweight='bold')
plt.show()

# Displaying the CSM plot for the best value of the K parameter for each dataset

# For the treatment  Dataset


scaler = StandardScaler()
X = scaler.fit_transform(treatmentPComps)

range_n_clusters = [4]

for n_clusters in range_n_clusters:
    fig, (ax1) = plt.subplots(1, 1)
    fig.set_size_inches(18, 7)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
    clusterer = KMeans(n_clusters=n_clusters, random_state=200)
    cluster_labels = clusterer.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    sample_silhouette_values = silhouette_samples(X, cluster_labels)
    y_lower = 10

    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10
    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])
    ax1.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.suptitle(("Silhouette analysis for KMeans clustering on the treatment dataset "
                  "with n_clusters = %d" % n_clusters), fontsize=14, fontweight='bold')

plt.show()

# Task 1, Part B
# Obtaining the optimal epsilon to be able to conduct the DBSCAN for each dataset
# For the DowJones Dataset

# Obtaining the optimal number for epsilon
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clustering)

neighbors = NearestNeighbors(n_neighbors=18)
neighbors_fit = neighbors.fit(X_scaled)
distances, indices = neighbors_fit.kneighbors(X_scaled)
distances = np.sort(distances, axis=0)
distances = distances[:, 1]
plt.figure(figsize=(10, 5))
plt.plot(distances)
plt.show()

eps = list(np.arange(1, 2.5, 0.05))

silhouette_coefficients = []
davis_bouldin_score = []

t = time.process_time()
for i in list(np.arange(1, 2.5, 0.2)):
    db = DBSCAN(eps=i, min_samples=18)
    db.fit(X_scaled)
    labels = db.labels_
    dbs = sklearn.metrics.davies_bouldin_score(X_scaled, labels)
    davis_bouldin_score.append(dbs)
    score = silhouette_score(X_scaled, db.labels_)
    silhouette_coefficients.append(score)
elapsed_time = time.process_time() - t
print("The time elapsed is:", elapsed_time)

plt.figure(figsize=(20, 5))
plt.subplot(1, 2, 1)
plt.title('Davis Bouldin Score as a function of epsilon')
plt.plot(list(np.arange(1, 2.5, 0.2)), davis_bouldin_score)
plt.xticks(list(np.arange(1, 2.5, 0.2)))
plt.xlabel("Epsilon")
plt.ylabel("Davis Bouldin Score")

plt.subplot(1, 2, 2)
plt.plot(list(np.arange(1, 2.5, 0.2)), silhouette_coefficients)
plt.title('Silhouette Coefficient as a function of no of epsilon')
plt.xticks(list(np.arange(1, 2.5, 0.2)))
plt.xlabel("Epsilon")
plt.ylabel("Silhouette Coefficient")

plt.show()

# For the Sales Dataset


# Obtaining the optimal number for epsilon
scaler = StandardScaler()
X_scaled = scaler.fit_transform(salesPComp)

neighbors = NearestNeighbors(n_neighbors=2)
neighbors_fit = neighbors.fit(X_scaled)
distances, indices = neighbors_fit.kneighbors(X_scaled)
distances = np.sort(distances, axis=0)
distances = distances[:, 1]
plt.figure(figsize=(10, 5))
plt.plot(distances)
plt.show()

silhouette_coefficients = []
davis_bouldin_score = []

t = time.process_time()
for i in list(np.arange(0.01, 0.05, 0.005)):
    db = DBSCAN(eps=i)
    db.fit(X_scaled)
    labels = db.labels_
    dbs = sklearn.metrics.davies_bouldin_score(X_scaled, labels)
    davis_bouldin_score.append(dbs)
    score = silhouette_score(X_scaled, db.labels_)
    silhouette_coefficients.append(score)
elapsed_time = time.process_time() - t
print("The time elapsed is:", elapsed_time)

plt.figure(figsize=(20, 5))
plt.subplot(1, 2, 1)
plt.title('Davis Bouldin Score as a function of epsilon')
plt.plot(list(np.arange(0.01, 0.05, 0.005)), davis_bouldin_score)
plt.xticks(list(np.arange(0.01, 0.05, 0.005)))
plt.xlabel("Epsilon")
plt.ylabel("Davis Bouldin Score")

plt.subplot(1, 2, 2)
plt.plot(list(np.arange(0.01, 0.05, 0.005)), silhouette_coefficients)
plt.title('Silhouette Coefficient as a function of no of epsilon')
plt.xticks(list(np.arange(0.01, 0.05, 0.005)))
plt.xlabel("Epsilon")
plt.ylabel("Silhouette Coefficient")

plt.show()

# For the facebook Dataset


# Obtaining the optimal number for epsilon
scaler = StandardScaler()
X_scaled = scaler.fit_transform(facebookPComps)

neighbors = NearestNeighbors(n_neighbors=6)
neighbors_fit = neighbors.fit(X_scaled)
distances, indices = neighbors_fit.kneighbors(X_scaled)
distances = np.sort(distances, axis=0)
distances = distances[:, 1]
plt.figure(figsize=(10, 5))
plt.plot(distances)
plt.show()

silhouette_coefficients = []
davis_bouldin_score = []

t = time.process_time()
for i in list(np.arange(0.01, 0.5, 0.05)):
    db = DBSCAN(eps=i, min_samples=6)
    db.fit(X_scaled)
    labels = db.labels_
    dbs = sklearn.metrics.davies_bouldin_score(X_scaled, labels)
    davis_bouldin_score.append(dbs)
    score = silhouette_score(X_scaled, db.labels_)
    silhouette_coefficients.append(score)
elapsed_time = time.process_time() - t
print("The time elapsed is:", elapsed_time)

plt.figure(figsize=(20, 5))
plt.subplot(1, 2, 1)
plt.title('Davis Bouldin Score as a function of epsilon')
plt.plot(list(np.arange(0.01, 0.5, 0.05)), davis_bouldin_score)
plt.xticks(list(np.arange(0.01, 0.5, 0.05)))
plt.xlabel("Epsilon")
plt.ylabel("Davis Bouldin Score")

plt.subplot(1, 2, 2)
plt.plot(list(np.arange(0.01, 0.5, 0.05)), silhouette_coefficients)
plt.title('Silhouette Coefficient as a function of no of epsilon')
plt.xticks(list(np.arange(0.01, 0.5, 0.05)))
plt.xlabel("Epsilon")
plt.ylabel("Silhouette Coefficient")

plt.show()

# For the treatment dataset


# Obtaining the optimal number for epsilon
scaler = StandardScaler()
X_scaled = scaler.fit_transform(treatmentPComps)

neighbors = NearestNeighbors(n_neighbors=6)
neighbors_fit = neighbors.fit(X_scaled)
distances, indices = neighbors_fit.kneighbors(X_scaled)
distances = np.sort(distances, axis=0)
distances = distances[:, 1]
plt.figure(figsize=(10, 5))
plt.plot(distances)
plt.show()

silhouette_coefficients = []
davis_bouldin_score = []

t = time.process_time()
for i in list(np.arange(0.3, 1.05, 0.05)):
    db = DBSCAN(eps=i, min_samples=6)
    db.fit(X_scaled)
    labels = db.labels_
    dbs = sklearn.metrics.davies_bouldin_score(X_scaled, labels)
    davis_bouldin_score.append(dbs)
    score = silhouette_score(X_scaled, db.labels_)
    silhouette_coefficients.append(score)
elapsed_time = time.process_time() - t
print("The time elapsed is:", elapsed_time)

plt.figure(figsize=(20, 5))
plt.subplot(1, 2, 1)
plt.title('Davis Bouldin Score as a function of epsilon')
plt.plot(list(np.arange(0.3, 1.05, 0.05)), davis_bouldin_score)
plt.xticks(list(np.arange(0.3, 1.05, 0.05)))
plt.xlabel("Epsilon")
plt.ylabel("Davis Bouldin Score")

plt.subplot(1, 2, 2)
plt.plot(list(np.arange(0.3, 1.05, 0.05)), silhouette_coefficients)
plt.title('Silhouette Coefficient as a function of no of epsilon')
plt.xticks(list(np.arange(0.3, 1.05, 0.05)))
plt.xlabel("Epsilon")
plt.ylabel("Silhouette Coefficient")

plt.show()

# Obtaining the 4 by 3 table


results = [['DowJones', 1.625, 1.6923249721, 0.390452298], ['Sales', 0.375, 0.88680364, 0.58727937],
           ['Facebook', 31.0625, 0.9638325, 0.520106853], ['Treatment', 1.55, 1.720287645, 0.6103995479629347]]
result = pd.DataFrame(results, columns=['Dataset', 'Time taken', 'Davis Bouldin Score', 'Cluster Silhouette Measure'])
print(result)

# Displaying the CSM plot for the best value of the epsilon parameter for each dataset

# For the DowJones Dataset


scaler = StandardScaler()
X = scaler.fit_transform(X_clustering)
fig, (ax1) = plt.subplots(1, 1)
fig.set_size_inches(18, 7)
ax1.set_xlim([0, 1])
ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
clusterer = db = DBSCAN(eps=2, min_samples=18)
cluster_labels = clusterer.fit_predict(X)
silhouette_avg = silhouette_score(X, cluster_labels)
sample_silhouette_values = silhouette_samples(X, cluster_labels)
y_lower = 10

for i in range(2):
    ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
    ith_cluster_silhouette_values.sort()
    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i
    color = cm.nipy_spectral(float(i) / n_clusters)
    ax1.fill_betweenx(np.arange(y_lower, y_upper),
                      0, ith_cluster_silhouette_values,
                      facecolor=color, edgecolor=color, alpha=0.7)
    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    y_lower = y_upper + 10
ax1.set_title("The silhouette plot for the various clusters.")
ax1.set_xlabel("The silhouette coefficient values")
ax1.set_ylabel("Cluster labels")
ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

ax1.set_yticks([])
ax1.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])

plt.suptitle(("Silhouette analysis for DBSCAN clustering on the DowJones dataset "
              "with epsilon = 2.0 "), fontsize=14, fontweight='bold')
plt.show()

# Displaying the CSM plot for the best value of the epsilon parameter for each dataset

# For the Sales Dataset


scaler = StandardScaler()
X = scaler.fit_transform(salesPComp)

scaler = StandardScaler()
X = scaler.fit_transform(X)
fig, (ax1) = plt.subplots(1, 1)
fig.set_size_inches(20, 7)
ax1.set_xlim([0, 1])
ax1.set_ylim([0, len(X)])
clusterer = DBSCAN(eps=0.015)
cluster_labels = clusterer.fit_predict(X)
silhouette_avg = silhouette_score(X, cluster_labels)
sample_silhouette_values = silhouette_samples(X, cluster_labels)
y_lower = 10

for i in range(15):
    ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
    ith_cluster_silhouette_values.sort()
    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i
    color = cm.nipy_spectral(float(i) / 15)
    ax1.fill_betweenx(np.arange(y_lower, y_upper),
                      0, ith_cluster_silhouette_values,
                      facecolor=color, edgecolor=color, alpha=0.7)
    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    y_lower = y_upper + 10
ax1.set_title("The silhouette plot for the various clusters.")
ax1.set_xlabel("The silhouette coefficient values")
ax1.set_ylabel("Cluster labels")
ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

ax1.set_yticks([])
ax1.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])

plt.suptitle(("Silhouette analysis for DBSCAN clustering on the Sales dataset "
              "with epsilon = 0.015"), fontsize=14, fontweight='bold')
plt.show()

# Displaying the CSM plot for the best value of the epsilon parameter for each dataset

# For the facebook Dataset


scaler = StandardScaler()
X = scaler.fit_transform(facebookPComps)

fig, (ax1) = plt.subplots(1, 1)
fig.set_size_inches(20, 7)
ax1.set_xlim([0, 1])
ax1.set_ylim([0, len(X)])
clusterer = DBSCAN(eps=0.21, min_samples=6)
cluster_labels = clusterer.fit_predict(X)
silhouette_avg = silhouette_score(X, cluster_labels)
sample_silhouette_values = silhouette_samples(X, cluster_labels)
y_lower = 10

for i in range(12):
    ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
    ith_cluster_silhouette_values.sort()
    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i
    color = cm.nipy_spectral(float(i) / 12)
    ax1.fill_betweenx(np.arange(y_lower, y_upper),
                      0, ith_cluster_silhouette_values,
                      facecolor=color, edgecolor=color, alpha=0.7)
    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    y_lower = y_upper + 10
ax1.set_title("The silhouette plot for the various clusters.")
ax1.set_xlabel("The silhouette coefficient values")
ax1.set_ylabel("Cluster labels")
ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

ax1.set_yticks([])
ax1.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])

plt.suptitle(("Silhouette analysis for DBSCAN clustering on the facebook dataset "
              "with epsilon = 0.21"), fontsize=14, fontweight='bold')
plt.show()

# Displaying the CSM plot for the best value of the epsilon parameter for each dataset

# For the treatment Dataset


scaler = StandardScaler()
X = scaler.fit_transform(treatmentPComps)

fig, (ax1) = plt.subplots(1, 1)
fig.set_size_inches(20, 7)
ax1.set_xlim([-2, 1])
ax1.set_ylim([0, len(X)])
clusterer = DBSCAN(eps=0.21, min_samples=6)
cluster_labels = clusterer.fit_predict(X)
silhouette_avg = silhouette_score(X, cluster_labels)
sample_silhouette_values = silhouette_samples(X, cluster_labels)
y_lower = 10

for i in range(1):
    ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
    ith_cluster_silhouette_values.sort()
    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i
    color = cm.nipy_spectral(float(i) / 12)
    ax1.fill_betweenx(np.arange(y_lower, y_upper),
                      0, ith_cluster_silhouette_values,
                      facecolor=color, edgecolor=color, alpha=0.7)
    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    y_lower = y_upper + 10
ax1.set_title("The silhouette plot for the various clusters.")
ax1.set_xlabel("The silhouette coefficient values")
ax1.set_ylabel("Cluster labels")
ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

ax1.set_yticks([])
ax1.set_xticks([-1, -0.5, 0, 0.2, 0.4, 0.6, 0.8, 1])

plt.suptitle(("Silhouette analysis for DBSCAN clustering on the treatment dataset "
              "with epsilon = 0.85"), fontsize=14, fontweight='bold')
plt.show()

# Task 1, Part C
# Obtaining the optimal number of clusters to be able to conduct the Agglomerative Clustering for each dataset
# For the DowJones Dataset


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clustering)

silhouette_coefficients = []
davis_bouldin_score = []

t = time.process_time()
for i in range(2, 11):
    db = AgglomerativeClustering(n_clusters=i)
    db.fit(X_scaled)
    labels = db.labels_
    dbs = sklearn.metrics.davies_bouldin_score(X_scaled, labels)
    davis_bouldin_score.append(dbs)
    score = silhouette_score(X_scaled, db.labels_)
    silhouette_coefficients.append(score)
elapsed_time = time.process_time() - t
print("The time elapsed is:", elapsed_time)

plt.figure(figsize=(20, 5))
plt.subplot(1, 2, 1)
plt.title('Davis Bouldin Score as a function of clusters')
plt.plot(range(2, 11), davis_bouldin_score)
plt.xticks(range(2, 11))
plt.xlabel("Clusters")
plt.ylabel("Davis Bouldin Score")

plt.subplot(1, 2, 2)
plt.plot(range(2, 11), silhouette_coefficients)
plt.title('Silhouette Coefficient as a function of no of clusters')
plt.xticks(range(2, 11))
plt.xlabel("Clusters")
plt.ylabel("Silhouette Coefficient")

plt.show()

# For the Sales Dataset


scaler = StandardScaler()
X_scaled = scaler.fit_transform(salesPComp)

silhouette_coefficients = []
davis_bouldin_score = []

t = time.process_time()
for i in range(2, 11):
    db = AgglomerativeClustering(n_clusters=i)
    db.fit(X_scaled)
    labels = db.labels_
    dbs = sklearn.metrics.davies_bouldin_score(X_scaled, labels)
    davis_bouldin_score.append(dbs)
    score = silhouette_score(X_scaled, db.labels_)
    silhouette_coefficients.append(score)
elapsed_time = time.process_time() - t
print("The time elapsed is:", elapsed_time)

plt.figure(figsize=(20, 5))
plt.subplot(1, 2, 1)
plt.title('Davis Bouldin Score as a function of clusters')
plt.plot(range(2, 11), davis_bouldin_score)
plt.xticks(range(2, 11))
plt.xlabel("Clusters")
plt.ylabel("Davis Bouldin Score")

plt.subplot(1, 2, 2)
plt.plot(range(2, 11), silhouette_coefficients)
plt.title('Silhouette Coefficient as a function of no of clusters')
plt.xticks(range(2, 11))
plt.xlabel("Clusters")
plt.ylabel("Silhouette Coefficient")

plt.show()

# For the facebook Dataset


scaler = StandardScaler()
X_scaled = scaler.fit_transform(facebookPComps)

silhouette_coefficients = []
davis_bouldin_score = []

t = time.process_time()
for i in range(2, 11):
    db = AgglomerativeClustering(n_clusters=i)
    db.fit(X_scaled)
    labels = db.labels_
    dbs = sklearn.metrics.davies_bouldin_score(X_scaled, labels)
    davis_bouldin_score.append(dbs)
    score = silhouette_score(X_scaled, db.labels_)
    silhouette_coefficients.append(score)
elapsed_time = time.process_time() - t
print("The time elapsed is:", elapsed_time)

plt.figure(figsize=(20, 5))
plt.subplot(1, 2, 1)
plt.title('Davis Bouldin Score as a function of epsilon')
plt.plot(range(2, 11), davis_bouldin_score)
plt.xticks(range(2, 11))
plt.xlabel("Clusters")
plt.ylabel("Davis Bouldin Score")

plt.subplot(1, 2, 2)
plt.plot(range(2, 11), silhouette_coefficients)
plt.title('Silhouette Coefficient as a function of no of epsilon')
plt.xticks(range(2, 11))
plt.xlabel("Clusters")
plt.ylabel("Silhouette Coefficient")

plt.show()

# For the treatment Dataset


scaler = StandardScaler()
X_scaled = scaler.fit_transform(treatmentPComps)

silhouette_coefficients = []
davis_bouldin_score = []

t = time.process_time()
for i in range(2, 11):
    db = AgglomerativeClustering(n_clusters=i)
    db.fit(X_scaled)
    labels = db.labels_
    dbs = sklearn.metrics.davies_bouldin_score(X_scaled, labels)
    davis_bouldin_score.append(dbs)
    score = silhouette_score(X_scaled, db.labels_)
    silhouette_coefficients.append(score)
elapsed_time = time.process_time() - t
print("The time elapsed is:", elapsed_time)

plt.figure(figsize=(20, 5))
plt.subplot(1, 2, 1)
plt.title('Davis Bouldin Score as a function of epsilon')
plt.plot(range(2, 11), davis_bouldin_score)
plt.xticks(range(2, 11))
plt.xlabel("Clusters")
plt.ylabel("Davis Bouldin Score")

plt.subplot(1, 2, 2)
plt.plot(range(2, 11), silhouette_coefficients)
plt.title('Silhouette Coefficient as a function of no of epsilon')
plt.xticks(range(2, 11))
plt.xlabel("Clusters")
plt.ylabel("Silhouette Coefficient")

plt.show()

# Obtaining the 4 by 3 table for the aggolmerative Clustering


results = [['DowJones', 1.2343, 1.07082473, 0.2651195], ['Sales', 0.34375, 0.3094922896, 0.75996619],
           ['Facebook', 43, 0.61793, 0.69166328], ['Treatment', 0.8594, 0.93382, 0.280028]]
result = pd.DataFrame(results, columns=['Dataset', 'Time taken', 'Davis Bouldin Score', 'Cluster Silhouette Measure'])
print(result)

# Displaying the CSM plot for the best value of K for the agglomerative clustering on each dataset

# For the DowJones Dataset


scaler = StandardScaler()
X = scaler.fit_transform(X_clustering)

range_n_clusters = [5]

for n_clusters in range_n_clusters:
    fig, (ax1) = plt.subplots(1, 1)
    fig.set_size_inches(18, 7)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
    clusterer = AgglomerativeClustering(n_clusters=n_clusters)
    cluster_labels = clusterer.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    sample_silhouette_values = silhouette_samples(X, cluster_labels)
    y_lower = 10

    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10
    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])
    ax1.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.suptitle(("Silhouette analysis for AgglomerativeClustering clustering on the dowjones dataset "
                  "with n_clusters = %d" % n_clusters), fontsize=14, fontweight='bold')
plt.show()

# Displaying the CSM plot for the best value of K for the agglomerative clustering on each dataset

# For the sales dataset


scaler = StandardScaler()
X = scaler.fit_transform(salesPComp)

range_n_clusters = [4]

for n_clusters in range_n_clusters:
    fig, (ax1) = plt.subplots(1, 1)
    fig.set_size_inches(18, 7)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
    clusterer = AgglomerativeClustering(n_clusters=n_clusters)
    cluster_labels = clusterer.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    sample_silhouette_values = silhouette_samples(X, cluster_labels)
    y_lower = 10

    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10
    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])
    ax1.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.suptitle(("Silhouette analysis for AgglomerativeClustering clustering on the sales dataset "
                  "with n_clusters = %d" % n_clusters), fontsize=14, fontweight='bold')
plt.show()

# Displaying the CSM plot for the best value of K for the agglomerative clustering on each dataset

# For the facebook dataset


scaler = StandardScaler()
X = scaler.fit_transform(facebookPComps)

range_n_clusters = [4]

for n_clusters in range_n_clusters:
    fig, (ax1) = plt.subplots(1, 1)
    fig.set_size_inches(18, 7)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
    clusterer = AgglomerativeClustering(n_clusters=n_clusters)
    cluster_labels = clusterer.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    sample_silhouette_values = silhouette_samples(X, cluster_labels)
    y_lower = 10

    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10
    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])
    ax1.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.suptitle(("Silhouette analysis for AgglomerativeClustering clustering on the facebook dataset "
                  "with n_clusters = %d" % n_clusters), fontsize=14, fontweight='bold')
plt.show()

# Displaying the CSM plot for the best value of K for the agglomerative clustering on each dataset

# For the treatment dataset


scaler = StandardScaler()
X = scaler.fit_transform(treatmentPComps)

range_n_clusters = [4]

for n_clusters in range_n_clusters:
    fig, (ax1) = plt.subplots(1, 1)
    fig.set_size_inches(18, 7)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
    clusterer = AgglomerativeClustering(n_clusters=n_clusters)
    cluster_labels = clusterer.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    sample_silhouette_values = silhouette_samples(X, cluster_labels)
    y_lower = 10

    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10
    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])
    ax1.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.suptitle(("Silhouette analysis for AgglomerativeClustering clustering on the DowJones dataset "
                  "with n_clusters = %d" % n_clusters), fontsize=14, fontweight='bold')
plt.show()
