import pandas as pd
data = pd.read_csv("dataset4clustering_student.csv")

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['address'] = le.fit_transform(data['address'])
data['famsize'] = le.fit_transform(data['famsize'])
data['Pstatus'] = le.fit_transform(data['Pstatus'])
data['Mjob'] = le.fit_transform(data['Mjob'])
data['Fjob'] = le.fit_transform(data['Fjob'])
data['guardian'] = le.fit_transform(data['guardian'])
data['reason'] = le.fit_transform(data['reason'])
data['school'] = le.fit_transform(data['school'])
data['sex'] = le.fit_transform(data['sex'])

data_features = data.drop(['schoolsup','famsup','paid','activities',
               'nursery','higher','internet','romantic'],  axis=1)

from sklearn.cluster import KMeans
wcss = []
for i in range(1, 14):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0,
                    max_iter=300,n_init = 10)
    kmeans.fit(data_features)
    wcss.append(kmeans.inertia_)
    
import matplotlib.pyplot as plt
plt.plot(range(1, 14), wcss, '-o')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

k_means = KMeans(
    n_clusters=3, init='k-means++',
    n_init=10, max_iter=300, random_state=0)
result = k_means.fit_predict(data_features)
print(result)
print(k_means.inertia_)

#Vẽ plot
cols = ['G1','G2','G3']
data_plot = pd.read_csv("dataset4clustering_student.csv", usecols = cols)

k_means_plot = KMeans(
    n_clusters=3, init='k-means++',
    n_init=10, max_iter=300, random_state=0)
result = k_means_plot.fit_predict(data_plot)
center = k_means_plot.cluster_centers_
fig =  plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(data_plot.iloc[:,2], data_plot.iloc[:,1], data_plot.iloc[:,0], alpha= 0.5,
               marker='o',c=k_means_plot.labels_, s=15)
plt.title('Clustering')
ax.set_xlabel('G3')
ax.set_ylabel('G2')
ax.set_zlabel('G1')
ax.scatter(center[:,2], center[:,1], center[:,0], marker='o',
               c='r', s=300)
plt.show()