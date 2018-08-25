def load_stop_words():
    stopwords_file = open('persian-stopwords.txt', 'r', encoding='utf8')
    stopwords = stopwords_file.read().split('\n')
    stopwords_file.close()
    return stopwords


def get_clusters(algorithm, excel_path):
    """
    :param algorithm: a function that get a path and return the clusters
    :param excel_path: a string
    :return: a list of Cluster objects
    """
    return algorithm(excel_path)
