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

import pandas as pd
from java.nio.file import Paths
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.search import IndexSearcher, BooleanClause, BooleanQuery, TermQuery
from org.apache.lucene.index import DirectoryReader, Term
import numpy as np
import random
from org.apache.lucene.analysis.fa import PersianAnalyzer

stop_words_address = 'incremental_stopwords.txt'
# stop_words_address = '../persian-stopwords.txt'

def load_stop_words():
    return [(x.strip()) for x in open(stop_words_address, 'r', encoding='utf8').read().split('\n')]


lucene.initVM(vmargs=['-Djava.awt.headless=true'])

data = pd.ExcelFile('../IrancellQA.xlsx')
preproc = data.parse('preprocessed')
raw = data.parse('Raw')
preproc = preproc.applymap(str)
raw = raw.applymap(str)
precols = preproc.columns
rawcols = raw.columns
soal = 0
javab = 1


def get_all_sentences():
    sentences = []
    for i in range(0, 14191):
        # for i in range(0, 1000):
        preA = preproc[precols[javab]][i]
        sentences.append(preA)
    return sentences


sw = load_stop_words()


class DocRepo:
    def __init__(self, indexes=None):

        # self.analyzer = StandardAnalyzer()

        # self.analyzer = PersianAnalyzer(StopFilter.makeStopSet(sw))
        # self.analyzer = PersianAnalyzer()
        self.analyzer = StopAnalyzer(Paths.get(stop_words_address))
        self.config = IndexWriterConfig(self.analyzer)
        self.index = RAMDirectory()
        if indexes is not None:
            self.loadIndex(indexes)
            print("is not none")
        else:
            self.loadIndex()

    def loadIndex(self, indexes=None):
        self.w = IndexWriter(self.index, self.config)
        if indexes is None:
            for i in range(0, 14191):
                self.addDocument(i)
        else:
            for i in indexes:
                self.addDocument(i)

    def addDocument(self, id):
        preQ = preproc[precols[soal]][id]
        rawQ = raw[rawcols[soal]][id]
        rawA = raw[rawcols[javab]][id]
        preA = preproc[precols[javab]][id]
        doc = Document()
        doc.add(TextField("rq", rawQ, Field.Store.YES))
        doc.add(TextField("pq", preQ, Field.Store.YES))
        doc.add(TextField("pa", preA, Field.Store.YES))
        doc.add(TextField("ra", rawA, Field.Store.YES))
        doc.add(StringField("id", str(id), Field.Store.YES))
        self.w.addDocument(doc)
        self.w.commit()

    def __del__(self):
        self.w.close()
        # print("died")

    def get_most_similar(self, sentence):
        # print('query string is',string)
        # q = QueryParser('pa', self.analyzer).parse(sentence)
        query_builder = BooleanQuery.Builder()
        for token in sentence.split(' '):
            if token not in sw:
                qtq = TermQuery(Term("pa", token))
                query_builder.add(BooleanClause(qtq, BooleanClause.Occur.SHOULD))
        q = query_builder.build()
        hitsPerPage = 2
        if self.index is None:
            print("its none dude")
            return
        reader = DirectoryReader.open(self.w)
        self.searcher = IndexSearcher(reader)
        simi = BM25Similarity(12.0, 1.0)
        self.searcher.setSimilarity(simi)

        docs = self.searcher.search(q, hitsPerPage)
        hits = docs.scoreDocs

        # print("Found " + str(len(hits)) + " hits.")
        if len(hits) > 0:
            mate = self.searcher.doc(hits[0].doc).get("id")
            print("found something. mate: ", mate, "- score : ", hits[0].score)
            reader.close()
            return hits[0], int(mate)
        else:
            reader.close()
            return None, -1


sentences = get_all_sentences()


def do_cluster(path, threshold):
    clusters = []
    # repo = DocRepo(path)

    random.shuffle(sentences)
    flags = []
    for i in range(0, len(sentences)):
        flags.append(-1)
    repo = DocRepo([])
    # reader = DirectoryReader.open(repo.w)
    # searcher = IndexSearcher(reader)

    print('number of sentences ', len(sentences))
    for senidx, sentence in enumerate(sentences):
        best_matching_cluster = -1
        # best_macthing_score = -1
        closest, mate = repo.get_most_similar(sentence)
        if (closest is not None) and (closest.score >= threshold):
            best_matching_cluster = flags[mate]
        if best_matching_cluster == -1:
            clusters.append([])
            clusters[-1].append(senidx)
            print(senidx, ' creates new cluster')
            flags[senidx] = len(clusters) - 1
        else:
            print(senidx, ' goes to cluster ', best_matching_cluster)
            clusters[best_matching_cluster].append(senidx)
            flags[senidx] = best_matching_cluster
        repo.addDocument(senidx)
    return clusters, repo


# repo = DocRepo()

#
# print(df2[cols2[1]][0])
# print((searcher.doc(0)).get('pa'))

res, repo = do_cluster('IrancellQA.xlsx', threshold=10)
print([len(cluster) for cluster in res])
print(len(res))
ones = [cl for cl in res if len(cl) == 1]
print('ones : ', len(ones))
with open('ones.txt', 'w', encoding='utf-8') as f:
    for one in ones:
        f.write(str(sentences[one[0]]))
        f.write("\n")
        # f.write("----------------------------------------")

# analyze = o()
# print(analyze.getDefaultStopSet())
# sw  = load_stop_words()
# print(sw)
