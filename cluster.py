import pyspark
import random
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from cleaner import datardd as data
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer,TfidfTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

def init_spark():
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    return spark

def create_df(seed):
    schema = StructType([StructField("Description", StringType()), StructField("Category", IntegerType())])
    rdd_tuples = data.map(lambda x: (x['d'], int(x['cscat'])))
    return init_spark().createDataFrame(rdd_tuples, schema)

def graph_by_category(pipeline, data):
    colormap = { 1: 'red', 8: 'green', 9: 'blue', 11: 'orange', 13: 'yellow', 14: 'magenta',  15: 'black'}
    texts = [a[0] for a in data]
    categories = [a[1] for a in data]
    category_colors = [colormap[a] for a in categories]

    matrix = pipeline.fit_transform(texts)

    X = matrix.todense()
    pca = PCA(n_components=2).fit(X)
    data2D = pca.transform(X)
    plt.scatter(data2D[:,0], data2D[:,1], c=category_colors)
    plt.show()

def generate_kmeans(k, pipeline, data):
    km = KMeans(init='k-means++', n_clusters=k, n_init=15)
    matrix = pipeline.fit_transform(data)
    km.fit(matrix)
    
    print("Top terms per cluster:")
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    terms = pipeline.get_feature_names()
    for i in range(k):
        top_words = [terms[ind] for ind in order_centroids[i, :10]]
        print("Cluster {}: {}".format(i+1, ' '.join(top_words)))
    graph_kmeans(km, matrix)

def graph_kmeans(km, matrix):
    X = matrix.todense()
    pca = PCA(n_components=2).fit(X)
    data2D = pca.transform(X)
    plt.scatter(data2D[:,0], data2D[:,1], c=km.labels_.astype(float))
    centers2D = pca.transform(km.cluster_centers_)
    markers =["o", "v", "^", "<", ">", "s", "p"]
    for i in range(0, len(centers2D)):
        plt.scatter(centers2D[i][0], centers2D[i][1], marker=markers[i], linewidths=2, c='r')
    plt.show()

df = create_df(10)
texts = data.map(lambda x: x['d']).collect()
pipeline = TfidfVectorizer(use_idf=False, norm='l1')
graph_by_category(pipeline, df.rdd.collect())
generate_kmeans(4, pipeline, texts)

