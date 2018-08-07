# -*- coding: utf-8 -*-
from xlrd import open_workbook
import gensim
from nltk.tokenize import RegexpTokenizer

wb = open_workbook('./QA-samples.xlsx')
sheet = wb.sheet_by_index(0)

answer_tokens_set = []
tokenizer = RegexpTokenizer(r'\w+')

stopwords_file = open('persian-stopwords.txt', 'r', encoding='utf8')
stopwords = stopwords_file.read().split('\n')
stopwords_file.close()

max_row = sheet.nrows
for row in range(1, 10):
    # q = str(sheet.cell(row, 0).value)
    a = str(sheet.cell(row, 1).value)
    tokens = tokenizer.tokenize(a)
    stopped_tokens = [word for word in tokens if word not in stopwords]
    answer_tokens_set.append(stopped_tokens)

dictionary = gensim.corpora.Dictionary(answer_tokens_set)
bag_of_word_matrix = [dictionary.doc2bow(tokens) for tokens in answer_tokens_set]
lda_model = gensim.models.LdaModel(bag_of_word_matrix, num_topics=10, id2word=dictionary, passes=20)
print(str(lda_model.print_topics(num_topics=10, num_words=10)).replace(', ', '\n\n'))
