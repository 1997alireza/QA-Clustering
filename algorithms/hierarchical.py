import pandas as pd
import numpy as np
from cluster import Cluster
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
# from scipy.cluster.hierarchy import ward, dendrogram
from sklearn.metrics.pairwise import euclidean_distances


def read_data(data_path):
    return pd.read_excel(data_path, sheet_name='preprocessed')


def load_stopwords():
    return [(x.strip()) for x in open('../persian-stopwords.txt', 'r').read().split('\n')]


def create_corpus(pandas_data):
    corpus = []
    for each_row in range(0, pandas_data.shape[0]):
        if (type(pandas_data.iat[each_row, 1]) is str):
            corpus.append(pandas_data.iat[each_row, 1])
    return corpus


# tfidf model
def create_transformed_model(corpus):
    count_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words=load_stopwords())
    docs_bag_of_words = count_vectorizer.fit_transform(corpus)
    docs_bag_of_words_feature_names = count_vectorizer.get_feature_names()
    docs_tfidf = TfidfTransformer().fit_transform(docs_bag_of_words)
    return docs_tfidf, docs_bag_of_words_feature_names


def create_2D_List(docs_tfidf):
    sentence = []


def create_hierarchical_model(n_clusters, linkage, affinity):
    return AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, affinity=affinity)
    # connectivity=connectivity
    # linkage_matrix = ward(distance)
    # print("here0")
    # fig, ax = plt.subplots(figsize=(30, 45))  # set size
    # ax = dendrogram(linkage_matrix, orientation="right", labels=titles);
    # print("here1")
    # plt.tick_params( \
    #     axis='x',  # changes apply to the x-axis
    #     which='both',  # both major and minor ticks are affected
    #     bottom='off',  # ticks along the bottom edge are off
    #     top='off',  # ticks along the top edge are off
    #     labelbottom='off')
    # # show plot with tight layout
    # plt.tight_layout()
    #
    # # AgglomerativeClustering(n_clusters=100, linkage=linkage_matrix, affinity='euclidean')
    # # plt.show()
    # # save figure as ward_clusters
    # plt.savefig('ward_clusters.png', dpi=200)
    # print("here2")
    # plt.close()
    # return type(ax)


def hierachical(data_path):
    clusters = []
    number_of_topics = 1
    pandas_data = read_data(data_path=data_path)
    corpus = create_corpus(pandas_data=pandas_data)
    docs_tfidf, _ = create_transformed_model(corpus[:300])
    hierarchical_model = create_hierarchical_model(n_clusters=number_of_topics, linkage='ward', affinity='euclidean')
    hierarchical_model.fit(docs_tfidf.toarray())
    return clusters


if __name__ == '__main__':
    hierachical("../QA-samples.xlsx")