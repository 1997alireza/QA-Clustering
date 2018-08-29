import lucene

from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.analysis.standard import StandardAnalyzer, StandardTokenizer
from org.apache.lucene.store import RAMDirectory
from org.apache.lucene.index import IndexWriter
from org.apache.lucene.index import IndexWriterConfig
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.document import Field
from org.apache.lucene.document import TextField
from org.apache.lucene.document import StringField
from org.apache.lucene.document import IntPoint
from org.apache.lucene.document import Document
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.analysis.core import StopAnalyzer
from org.apache.lucene.analysis import StopwordAnalyzerBase
from org.apache.lucene.analysis import StopFilter
from org.apache.lucene.search.similarities import BM25Similarity
from org.apache.lucene.search.similarities import ClassicSimilarity
from java.nio.file import Paths
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.search import IndexSearcher, BooleanClause, BooleanQuery, TermQuery
from org.apache.lucene.index import DirectoryReader, Term
import random
from org.apache.lucene.analysis.fa import PersianAnalyzer
import pandas as pd
from random import randint
import matplotlib.pyplot as plt
import numpy as np


class Config:
    stop_words_address = 'incremental_stopwords.txt'
    # stop_words_address = '../persian-stopwords.txt'
    k1 = 3.0
    b = 0.75
    threshold = 9.0
    train_size = 10000
    test_size = 4000


lucene.initVM(vmargs=['-Djava.awt.headless=true'])


def load_stop_words():
    return [(x.strip()) for x in open(Config.stop_words_address, 'r', encoding='utf8').read().split('\n')]


sw = load_stop_words()


# def load_data(path):
#     global data, preproc, raw, precols, rawcols, soal, javab
#     global records_numb, sw
#     global answers_test, questions_test, answers_train, questions_train
#
# data = pd.ExcelFile(path)
#     preproc = data.parse('preprocessed')
#     raw = data.parse('Raw')
#     preproc = preproc.applymap(str)
#     raw = raw.applymap(str)
#     precols = preproc.columns
#     rawcols = raw.columns
#     soal = 0
#     javab = 1
#     records_numb = len(preproc[precols[javab]])
#     sw = load_stop_words()
#     answers_train, questions_train = [], []
#     answers_test, questions_test = [], []
#     for i in range(0, Config.train_size):
#         a_pre = preproc[precols[javab]][i]
#         q_pre = preproc[precols[soal]][i]
#         answers_train.append(a_pre)
#         questions_train.append(q_pre)
#     for i in range(1, Config.test_size):
#         i += Config.train_size
#         a_pre = preproc[precols[javab]][i]
#         q_pre = preproc[precols[soal]][i]
#         answers_test.append(a_pre)
#         questions_test.append(q_pre)


# def evaluate():
#     print("Loading Data...")
#     load_data('../IrancellQA.xlsx')
#     print("Data Was Loaded")
#     print("Clustering... ")
#     clusters, repo = do_cluster(Config.threshold)
#     print("Clustering Done")
#     global answers_test, questions_test, answers_train, questions_train, flags
#     numbers = []
#     for i, q in enumerate(questions_test):
#         near = repo.get_nearest_question(q)
#         if near is not None:
#             # print("heyy")
#             clus = flags[int(near)]
#             answer_id = clusters[clus][randint(0, len(clusters[clus]) - 1)]
#             numbers.append(editDistance(answers_train[answer_id], answers_test[i]))
#     plt.hist(numbers, bins=range(0, 600))
#     plt.title("Histogram")
#     plt.xlabel("Value")
#     plt.ylabel("Frequency")
#     plt.show()


class DocRepo:
    def __init__(self):
        # self.analyzer = StandardAnalyzer()
        # self.analyzer = PersianAnalyzer(StopFilter.makeStopSet(sw))
        # self.analyzer = PersianAnalyzer()
        self.analyzer = StopAnalyzer(Paths.get(Config.stop_words_address))
        self.config = IndexWriterConfig(self.analyzer)
        self.index = RAMDirectory()
        self.w = IndexWriter(self.index, self.config)

    def addDocument(self, id):
        global answers_train
        # preQ = preproc[precols[soal]][id]
        # rawQ = raw[rawcols[soal]][id]
        # rawA = raw[rawcols[javab]][id]
        # preA = preproc[precols[javab]][id]
        preA = answers_train[id]
        doc = Document()
        # doc.add(TextField("rq", rawQ, Field.Store.YES))
        # doc.add(TextField("pq", preQ, Field.Store.YES))
        doc.add(TextField("pa", preA, Field.Store.YES))
        # doc.add(TextField("ra", rawA, Field.Store.YES))
        doc.add(StringField("id", str(id), Field.Store.YES))
        self.w.addDocument(doc)
        self.w.commit()

    def __del__(self):
        self.w.close()

    # def get_nearest_question(self, question):
    #     query_builder = BooleanQuery.Builder()
    #     for token in question.split(' '):
    #         if token not in sw:
    #             qtq = TermQuery(Term("pq", token))
    #             query_builder.add(BooleanClause(qtq, BooleanClause.Occur.SHOULD))
    #     q = query_builder.build()
    #     hitsPerPage = 2
    #     reader = DirectoryReader.open(self.w)
    #     self.searcher = IndexSearcher(reader)
    #     simi = BM25Similarity(Config.k1, Config.b)
    #     # simi = ClassicSimilarity()
    #     self.searcher.setSimilarity(simi)
    #
    #     docs = self.searcher.search(q, hitsPerPage)
    #     hits = docs.scoreDocs
    #     if len(hits) > 0:
    #         return (self.searcher.doc(hits[0].doc)).get('id')

    def get_most_similar(self, sentence, do_log=False):
        # print('query string is',string)
        # q = QueryParser('pa', self.analyzer).parse(sentence)
        query_builder = BooleanQuery.Builder()
        for token in sentence.split(' '):
            if token not in sw:
                qtq = TermQuery(Term("pa", token))
                query_builder.add(BooleanClause(qtq, BooleanClause.Occur.SHOULD))
        q = query_builder.build()
        hitsPerPage = 2
        reader = DirectoryReader.open(self.w)
        self.searcher = IndexSearcher(reader)
        simi = BM25Similarity(Config.k1, Config.b)
        # simi = ClassicSimilarity()
        self.searcher.setSimilarity(simi)

        docs = self.searcher.search(q, hitsPerPage)
        hits = docs.scoreDocs

        # print("Found " + str(len(hits)) + " hits.")
        if len(hits) > 0:
            mate = self.searcher.doc(hits[0].doc).get("id")
            if do_log:
                print("found something. mate: ", mate, "- score : ", hits[0].score)
            return hits[0], int(mate)
        else:
            return None, -1


def do_cluster(threshold, do_log=False):
    global answers_train
    clusters = []
    # repo = DocRepo(path)
    # random.shuffle(answers_train)
    global flags
    flags = []
    for i in range(0, len(answers_train)):
        flags.append(-1)
    repo = DocRepo()
    scores = []
    if do_log:
        print('number of sentences ', len(answers_train))
    for senidx, sentence in enumerate(answers_train):
        best_matching_cluster = -1
        closest, mate = repo.get_most_similar(sentence, do_log)
        if closest is not None:
            scores.append(closest.score)
        if (closest is not None) and (closest.score >= threshold):
            best_matching_cluster = flags[mate]
        if best_matching_cluster == -1:
            clusters.append([])
            clusters[-1].append(senidx)
            if do_log:
                print(senidx, ' creates new cluster')
            flags[senidx] = len(clusters) - 1
        else:
            if do_log:
                print(senidx, ' goes to cluster ', best_matching_cluster)
            clusters[best_matching_cluster].append(senidx)
            flags[senidx] = best_matching_cluster
        repo.addDocument(senidx)

    # x = range(0, len(scores))
    # plt.scatter(x, scores)
    # plt.show()
    print(np.mean(scores))
    print(np.var(scores))

    return clusters, repo


from cluster import Cluster


def incremental(train_records):
    # load_data(path)
    global answers_train
    answers_train = [rec.a_pre for rec in train_records]

    res, repo = do_cluster(Config.threshold)
    cluss = []
    for cl in res:
        cll = Cluster("not implemented yet")
        for numb in cl:
            cll.records.append(train_records[numb])
            # cll.add_doc((numb, answers_train[numb]))
        cluss.append(cll)
    return cluss


def test(train_records, do_log):
    import os
    # load_data('../IrancellQA.xlsx')
    global answers_train
    answers_train = [rec.a_pre for rec in train_records]
    res, repo = do_cluster(Config.threshold, do_log)
    i = 0
    os.makedirs("clusters")
    print(len(res))
    print(len([re for re in res if len(re) == 1]))
    ones = [cl for cl in res if len(cl) == 1]
    for cl in res:
        i += 1
        with open("clusters/" + str(i) + ".txt", 'w', encoding='utf-8') as f:
            for number in cl:
                if number not in ones:
                    f.write(answers_train[number])
                    f.write("\n--------------------------------\n")

    with open('ones.txt', 'w', encoding='utf-8') as f:
        for one in ones:
            f.write(str(answers_train[one[0]]))
            f.write("\n--------------------------\n")
    print([len(re) for re in res])

def perform_test():
    from main import read_data, make_records, divide_train_test

    df_pre, df_raw = read_data(data_path="../IrancellQA.xlsx")
    cor = make_records(df_pre, df_raw)
    train_records, test_records = divide_train_test(cor, 0.9)
    test(train_records, True)
# perform_test()