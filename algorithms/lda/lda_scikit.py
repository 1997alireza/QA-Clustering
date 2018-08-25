import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation
from cluster import Cluster


def read_data(data_path):
    df = pd.read_excel(data_path, sheet_name='preprocessed')
    return df


def make_corpus(pandas_data):
    corpus = []
    for each_row in range(0, pandas_data.shape[0]):
        if (type(pandas_data.iat[each_row, 0]) is str):
            corpus.append(pandas_data.iat[each_row, 0])
    return corpus


# Get the docs & make Bag of Words of them
def create_bow(pandas_data):
    # docs = list(df.iloc[0:, 0])
    corpus = make_corpus(pandas_data)
    stop_words = load_stopwords()
    count_vectorizer = CountVectorizer(stop_words=stop_words)
    docs_bag_of_words = count_vectorizer.fit_transform(corpus)
    docs_bag_of_words_feature_names = count_vectorizer.get_feature_names()
    docs_bag_of_words_tfidf = TfidfTransformer().fit_transform(docs_bag_of_words)

    return docs_bag_of_words, docs_bag_of_words_feature_names, docs_bag_of_words_tfidf


def load_stopwords():
    return [(x.strip()) for x in open('persian-stopwords.txt', 'r').read().split('\n')]


def lda_scikit(path):
    df = read_data(path)
    bow, bow_feature_names, bow_tfidf = create_bow(df)
    lda = LatentDirichletAllocation(n_components=60, learning_method='batch').fit(bow)
    topic_to_docs = lda.transform(bow)

    clusters = []
    no_top_documents = 10000
    for topic_idx, topic in enumerate(lda.components_):
        cluster = Cluster(topic)
        top_doc_indices = np.argsort(topic_to_docs[:, topic_idx])[::-1][0:no_top_documents]
        for doc_index in top_doc_indices:
            cluster.add_doc(df.iloc[doc_index, 0])
        clusters.append(cluster)

    return clusters
