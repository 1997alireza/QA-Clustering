def load_stop_words():
    stopwords_file = open('persian-stopwords.txt', 'r', encoding='utf8')
    stopwords = stopwords_file.read().split('\n')
    stopwords_file.close()
    return stopwords


def get_clusters(algorithm, corpus, number_of_clusters):
    """
    :param algorithm: a function that get a path and return the clusters
    :param corpus: an array of documents
    :param number_of_clusters: number of clusters that should be make
    :return: a list of Cluster objects
    """
    return algorithm(corpus, number_of_clusters)


def make_corpus(records):
    return [r.a_pre for r in records]
