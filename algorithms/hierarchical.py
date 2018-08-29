import numpy as np
from cluster import Cluster
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram

from tools import load_stop_words, make_corpus


# tfidf model
def create_transformed_model(corpus):
    count_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words=load_stop_words())
    docs_bag_of_words = count_vectorizer.fit_transform(corpus)
    docs_bag_of_words_feature_names = count_vectorizer.get_feature_names()
    docs_tfidf = TfidfTransformer().fit_transform(docs_bag_of_words)
    return docs_tfidf, docs_bag_of_words_feature_names


# function to draw hierarchical model
def plot_dendrogram(model, **kwargs):
    # Children of hierarchical clustering
    children = model.children_
    # Distances between each pair of children
    # Since we don't have this information, we can use a uniform one for plotting
    distance = np.arange(children.shape[0])

    # The number of observations contained in each cluster level
    no_of_observations = np.arange(2, children.shape[0] + 2)

    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


def create_cluster_members(labels, records, number_of_topics):
    clusters = []
    for i in range(number_of_topics):
        cluster = Cluster("")
        for j in range(len(labels)):
            if labels[j] == i:
                cluster.records.append(records[j])
        cluster.title = cluster.records[0].a_raw
        clusters.append(cluster)

    return clusters


def create_hierarchical_model(n_clusters, linkage, affinity):
    return AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, affinity=affinity)


def hierarchical(records):
    corpus = make_corpus(records=records)
    number_of_topics = 10
    docs_tfidf, _ = create_transformed_model(corpus)
    hierarchical_model = create_hierarchical_model(n_clusters=number_of_topics, linkage='ward', affinity='euclidean')
    model = hierarchical_model.fit(docs_tfidf.toarray())
    labels = model.labels_
    print(labels)
    clusters = create_cluster_members(labels=labels, records=records, number_of_topics=number_of_topics)
    plt.title('Hierarchical Clustering Dendrogram')
    plot_dendrogram(model, labels=model.labels_)
    plt.show()
    return clusters
