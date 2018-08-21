import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation


def read_data(data_path):
    df = pd.read_csv(data_path)
    df.fillna(' ', inplace=True)
    return df

# Get the docs & make Bag of Words of them
def create_bow(df):
    docs =list( df.iloc[0:,0])

    count_vectorizer = CountVectorizer()
    docs_bag_of_words = count_vectorizer.fit_transform(docs)
    docs_bag_of_words_feature_names = count_vectorizer.get_feature_names()
    docs_bag_of_words_tfidf = TfidfTransformer().fit_transform(docs_bag_of_words)
    
    return docs_bag_of_words, docs_bag_of_words_feature_names, docs_bag_of_words_tfidf


def lda_scikit(path):
    df = read_data(data_path)

    bow, bow_feature_names, bow_tfidf = create_bow(df)

    lda = LatentDirichletAllocation(n_components =60, learning_method='batch').fit(bow)
    topic_to_docs = lda.transform(bow)

    return topic_to_docs
