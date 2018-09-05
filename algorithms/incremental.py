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
from cluster import Cluster

# from main import read_data, make_records, divide_train_test


class Config:
    # stop_words_address = 'incremental_stopwords.txt'
    stop_words_address = 'persian-stopwords.txt'
    k1 = 1.2
    b = 0.75
    threshold = 17.0


def load_stop_words():
    return [(x.strip()) for x in open(Config.stop_words_address, 'r', encoding='utf8').read().split('\n')]


sw = load_stop_words()

lucene.initVM(vmargs=['-Djava.awt.headless=true'])


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
        preA = answers_train[id]
        doc = Document()
        doc.add(TextField("pa", preA, Field.Store.YES))
        doc.add(StringField("id", str(id), Field.Store.YES))
        self.w.addDocument(doc)
        self.w.commit()

    def __del__(self):
        self.w.close()

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


def incremental(train_records, num_clusters):
    # load_data(path)
    global answers_train
    answers_train = [rec.a_pre for rec in train_records]

    res, repo = do_cluster(Config.threshold)
    res = [cl for cl in res if len(cl) > 1]
    cluss = []
    for cl in res:
        cll = Cluster(train_records[cl[0]])
        for numb in cl:
            # cll.records.append(train_records[numb])
            cll.add_record(train_records[numb])
            # cll.add_doc((numb, answers_train[numb]))
        cluss.append(cll)
    return cluss


def test(train_records, do_log):
    import os
    # load_data('../IrancellQA.xlsx')
    global answers_train
    answers_train = [rec.a_pre for rec in train_records]
    res, repo = do_cluster(Config.threshold, do_log)
    ones = [cl for cl in res if len(cl) == 1]
    res = [cl for cl in res if len(cl) > 1]

    i = 0
    os.makedirs("clusters")
    print('number of clusters :', len(res))
    print('removed records :', len(ones))
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



