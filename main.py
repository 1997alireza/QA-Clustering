# -*- coding: utf-8 -*-
from xlrd import open_workbook
import gensim
from nltk.tokenize import RegexpTokenizer
import matplotlib.pyplot as plt

# q? Does lda model get TFIDF matrix too?


def load_stop_words():
    stopwords_file = open('persian-stopwords.txt', 'r', encoding='utf8')
    stopwords = stopwords_file.read().split('\n')
    stopwords_file.close()
    return stopwords


def get_tokens(sheet):
    tokens_set = []
    tokenizer = RegexpTokenizer(r'\w+')
    stopwords = load_stop_words()
    for row in range(0, sheet.nrows):
        a = str(sheet.cell(row, 1).value).replace('ي', 'ی')
        tokens = tokenizer.tokenize(a)
        stopped_tokens = [word for word in tokens if word not in stopwords]
        tokens_set.append(stopped_tokens)
    return tokens_set


def get_bow_matrix(token_set, dictionary):
    return [dictionary.doc2bow(tokens) for tokens in token_set]


def get_printable_answers_best_topic(model, train_bow_matrix, sheet):
    res = ""
    for row in range(0, len(train_bow_matrix)):
        res += "Answer: " + str(sheet.cell(row, 1).value) + "\n\n"
        topic_possibilities = model[train_bow_matrix[row]]
        res += str(topic_possibilities) + "\n"
        best_psb = -1
        best_topic = -1
        for pair in topic_possibilities:
            if pair[1] > best_psb:
                best_topic = pair[0]
                best_psb = pair[1]
        res += "Best topic: " + str(best_topic) + "\n\n------------------------\n"
    return res


def make_lda_model(train_bow_matrix, test_bow_matrix, num_topics, id2word, sheet, tfidf_model=False, get_log=False):
    train_matrix = get_tfidf_matrix(train_bow_matrix, id2word) if tfidf_model else train_bow_matrix
    test_matrix = get_tfidf_matrix(test_bow_matrix, id2word) if tfidf_model else test_bow_matrix

    lda_model = gensim.models.LdaModel(corpus=train_matrix, num_topics=num_topics, id2word=id2word, passes=20)
    perplexity = lda_model.log_perplexity(test_matrix)
    if get_log:
        log = get_printable_answers_best_topic(lda_model, train_matrix, sheet)
        return log, perplexity
    else:
        return perplexity


def get_tfidf_matrix(bow_matrix, id2word):
    tfidf_model = gensim.models.TfidfModel(corpus=bow_matrix, id2word=id2word, normalize=True)
    return [tfidf_model[bow] for bow in bow_matrix]


def calculate_perplexities(num_topics_range, tfidf_model=False, show_graph=False):
    print("---- calculate perplexities started! ----")
    perplexities = []
    topics_numbers = []
    for num_topics in num_topics_range:
        topics_numbers.append(num_topics)
        perplexities.append(make_lda_model(train_bow_matrix, test_bow_matrix, num_topics, dictionary, sheet,
                                           tfidf_model))
    if show_graph:
        plt.plot(topics_numbers, perplexities)
        plt.show()
    return perplexities


train_bow_matrix, test_bow_matrix, dictionary, sheet = [None] * 4


def main():
    global train_bow_matrix, test_bow_matrix, dictionary, sheet
    wb = open_workbook('./QA-samples.xlsx')
    sheet = wb.sheet_by_index(1)
    tokens_set = get_tokens(sheet)
    set_length = len(tokens_set)
    train_percent = 0.7
    train_tokens_set = tokens_set[:int(set_length*train_percent)]
    test_tokens_set = tokens_set[int(set_length*train_percent):]

    dictionary = gensim.corpora.Dictionary(tokens_set)
    train_bow_matrix = get_bow_matrix(train_tokens_set, dictionary)
    test_bow_matrix = get_bow_matrix(test_tokens_set, dictionary)

    perplexities = calculate_perplexities(range(60, 67), show_graph=True)
    print(perplexities)

    # log, _ = make_lda_model(train_bow_matrix, test_bow_matrix, 10, dictionary, sheet, True, True)
    # print(log)


if __name__ == "__main__":
    main()
