# -*- coding: utf-8 -*-
# from xlrd import open_workbook
import gensim
from nltk.tokenize import RegexpTokenizer
from tools import load_stop_words
from cluster import Cluster


def get_tokens(records):
    tokens_set = []
    tokenizer = RegexpTokenizer(r'\w+')
    stopwords = load_stop_words()
    for row in range(len(records)):
        # a = str(sheet.cell(row, 1).value).replace('ي', 'ی')
        tokens = tokenizer.tokenize(records[row].a_raw)
        stopped_tokens = [word for word in tokens if word not in stopwords]
        tokens_set.append(stopped_tokens)
    return tokens_set


def get_bow_matrix(token_set, dictionary):
    return [dictionary.doc2bow(tokens) for tokens in token_set]


def get_clusters(model, train_bow_matrix, records):
    clusters = []
    for t in range(0, model.num_topics):
        clusters.append(Cluster(model.show_topic(t)[0][0]))

    for row in range(0, len(train_bow_matrix)):
        topic_possibilities = model[train_bow_matrix[row]]
        best_psb = -1
        best_topic = -1
        for pair in topic_possibilities:
            if pair[1] > best_psb:
                best_topic = pair[0]
                best_psb = pair[1]
        clusters[best_topic].records.append(records[row])
    return clusters


def make_lda_model(train_bow_matrix, test_bow_matrix, num_topics, id2word, records, tfidf_model=False,
                   get_perplexity=False):
    train_matrix = get_tfidf_matrix(train_bow_matrix, id2word) if tfidf_model else train_bow_matrix
    test_matrix = get_tfidf_matrix(test_bow_matrix, id2word) if tfidf_model else test_bow_matrix

    lda_model = gensim.models.LdaModel(corpus=train_matrix, num_topics=num_topics, id2word=id2word, passes=20)
    clusters = get_clusters(lda_model, train_matrix, records)
    if get_perplexity:
        return clusters, lda_model.log_perplexity(test_matrix)
    else:
        return clusters


def get_tfidf_matrix(bow_matrix, id2word):
    tfidf_model = gensim.models.TfidfModel(corpus=bow_matrix, id2word=id2word, normalize=True)
    return [tfidf_model[bow] for bow in bow_matrix]


# def calculate_perplexities(num_topics_range, tfidf_model=False, show_graph=False):
#     start_range = num_topics_range.start
#     end_range = num_topics_range.stop
#     print("---- calculate perplexities started! ----")
#     log_perplexities = []
#     topics_numbers = []
#     # for num_topics in num_topics_range:
#     num_topics = start_range
#     while num_topics < end_range:
#         print(num_topics, "\n")
#         topics_numbers.append(num_topics)
#         log_perplexities.append(make_lda_model(train_bow_matrix, test_bow_matrix, num_topics, dictionary, sheet,
#                                                tfidf_model))
#         if num_topics % 2 == 0:
#             num_topics += int(num_topics / 2)
#         else:
#             num_topics += int(num_topics / 2) + 1
#     topics_numbers.append(end_range)
#     log_perplexities.append(make_lda_model(train_bow_matrix, test_bow_matrix, end_range, dictionary, sheet,
#                                            tfidf_model))
#
#     if show_graph:
#         plt.plot(topics_numbers, log_perplexities)
#         plt.show()
#     return log_perplexities


train_bow_matrix, test_bow_matrix, dictionary = [None] * 3


def lda_gensim(records, number_of_clusters):
    global train_bow_matrix, test_bow_matrix, dictionary
    tokens_set = get_tokens(records)
    set_length = len(tokens_set)
    train_percent = 1.0
    train_tokens_set = tokens_set[:int(set_length * train_percent)]
    test_tokens_set = tokens_set[int(set_length * train_percent):]

    dictionary = gensim.corpora.Dictionary(tokens_set)
    train_bow_matrix = get_bow_matrix(train_tokens_set, dictionary)
    test_bow_matrix = get_bow_matrix(test_tokens_set, dictionary)

    # log_perplexities = calculate_perplexities(range(1, 5000), show_graph=True, tfidf_model=True)
    # print(log_perplexities)

    return make_lda_model(train_bow_matrix, test_bow_matrix, number_of_clusters, dictionary, records, tfidf_model=False)
