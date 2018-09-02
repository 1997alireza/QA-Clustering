import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation
from cluster import LDACluster
from tools import load_stop_words, make_corpus


# Get the docs & make Bag of Words of them
def create_bow(corpus):
    stop_words = load_stop_words()
    count_vectorizer = CountVectorizer(stop_words=stop_words)
    docs_bag_of_words = count_vectorizer.fit_transform(corpus)
    docs_bag_of_words_feature_names = count_vectorizer.get_feature_names()
    docs_bag_of_words_tfidf = TfidfTransformer().fit_transform(docs_bag_of_words)

    return docs_bag_of_words, docs_bag_of_words_feature_names, docs_bag_of_words_tfidf


def lda_scikit(records, number_of_clusters):
    corpus = make_corpus(records=records)
    bow, bow_feature_names, bow_tfidf = create_bow(corpus)
    lda = LatentDirichletAllocation(n_components=number_of_clusters, learning_method='online').fit(bow)
    topic_to_docs = lda.transform(bow)

    clusters = []
    no_top_documents = 10
    no_top_words = 10

    # print(topic_to_docs.shape)
    # sum_ = 0
    # for i in range(0,150):
    #     sum_ = sum_ + topic_to_docs[0,i]
    # print(sum_)

    for topic_idx, topic in enumerate(lda.components_):
        title = [bow_feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
        cluster = LDACluster(title[0])
        clusters.append(cluster)

    for doc_index in range(len(records)):
        possibility = 0
        best_cluster_index = 0 
        for topic_index in range(number_of_clusters):
            if(topic_to_docs[doc_index, topic_index] > possibility):
                possibility = topic_to_docs[doc_index, topic_index]
                best_cluster_index = topic_index
        clusters[best_cluster_index].add_record_psb(records[doc_index], possibility)


    # for topic_idx, topic in enumerate(lda.components_):
    #     title = [bow_feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
    #     cluster = LDACluster(title[0])
    #     top_doc_indices = np.argsort(topic_to_docs[:, topic_idx])[::-1][0:no_top_documents]
    #     for doc_index in top_doc_indices:
    #         cluster.add_record(records[doc_index])
    #     clusters.append(cluster)

    return clusters
