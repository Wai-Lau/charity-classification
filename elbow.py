from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from cleaner import datardd as data

def elbow_graph(data, pipeline):
    matrix = pipeline.fit_transform(data)
    wcss = []
    for i in range(4,8):
        kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
        kmeans.fit(matrix)
        wcss.append(kmeans.inertia_)
    plt.plot(range(4, 8),wcss, 'purple')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.savefig('elbow_purple.png')
    plt.show()

texts = data.map(lambda x: x['d']).collect()
elbow_graph(texts, TfidfVectorizer(use_idf=False, norm='l1'))