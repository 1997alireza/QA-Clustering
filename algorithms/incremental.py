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
import pandas as pd
from java.nio.file import Paths
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.search import IndexSearcher, BooleanClause, BooleanQuery, TermQuery
from org.apache.lucene.index import DirectoryReader, Term
import numpy as np
import random
from org.apache.lucene.analysis.fa import PersianAnalyzer


# from '../../tools' import load_stop_words
# import '../../'tools as tols

def load_stop_words():
    stopwords_file = open('persian-stopwords.txt', 'r', encoding='utf8')
    stopwords = stopwords_file.read().split('\n')
    stopwords_file.close()
    return stopwords


lucene.initVM(vmargs=['-Djava.awt.headless=true'])

data = pd.ExcelFile('IrancellQA.xlsx')
df1 = data.parse('preprocessed')
df2 = data.parse('Raw')
df1 = df1.applymap(str)
df2 = df2.applymap(str)
cols1 = df1.columns
cols2 = df2.columns


def get_all_sentences():
    sentences = []
    for i in range(0, 14191):
        # for i in range(0, 1000):
        rawA = df2[cols2[1]][i]
        sentences.append(rawA)
    return sentences


class DocRepo:
    def __init__(self, indexes=None):

        # self.analyzer = StandardAnalyzer()
        sw = load_stop_words()

        # self.analyzer = PersianAnalyzer(StopFilter.makeStopSet(sw))
        self.analyzer = StopAnalyzer(Paths.get('persian-stopwords.txt'))
        self.config = IndexWriterConfig(self.analyzer)
        self.index = RAMDirectory()
        if indexes is not None:
            self.loadIndex(indexes)
            print("is not none")
        else:
            self.loadIndex()
        # print("hiiiiiii")

    def loadIndex(self, indexes=None):
        # print("loading")
        self.w = IndexWriter(self.index, self.config)
        if indexes is None:
            for i in range(0, 14191):
                # rawQ = df1[cols1[0]][i]
                # rawA = df2[cols2[0]][i]
                # preQ = df1[cols1[1]][i]
                # self.addDoc(i,rawQ, preQ, rawA)
                self.addDocument(i)
        else:
            for i in indexes:
                # rawQ = df1[cols1[0]][i]
                # rawA = df2[cols2[0]][i]
                # preQ = df1[cols1[1]][i]
                # self.addDoc(i,rawQ, preQ, rawA)
                self.addDocument(i)

    def addDoc(self, id, rawQuestion, preprocessedQuestion, RawAnswer):
        doc = Document()
        doc.add(TextField("rq", rawQuestion, Field.Store.YES))
        doc.add(TextField("pq", preprocessedQuestion, Field.Store.YES))
        doc.add(TextField("ra", RawAnswer, Field.Store.YES))
        doc.add(StringField("id", str(id), Field.Store.YES))
        self.w.addDocument(doc)
        self.w.commit()

    def addDocument(self, id):
        rawQ = df1[cols1[0]][id]
        rawA = df2[cols2[0]][id]
        preA = df2[cols2[1]][id]
        preQ = df1[cols1[1]][id]
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

    def get_most_similar(self, string):
        # print('query string is',string)
        # q = QueryParser(f = 'ra', a = self.analyzer).parse(string)
        query_builder = BooleanQuery.Builder()
        for q_word in string.split(' '):
            qtq = TermQuery(Term("pa", q_word))
            query_builder.add(BooleanClause(qtq, BooleanClause.Occur.SHOULD))
        q = query_builder.build()
        hitsPerPage = 2
        if self.index is None:
            print("its none dude")
        reader = DirectoryReader.open(self.w)
        searcher = IndexSearcher(reader)

        docs = searcher.search(q, hitsPerPage)
        hits = docs.scoreDocs

        # print("Found " + str(len(hits)) + " hits.")
        if len(hits) > 0:
            mate = searcher.doc(hits[0].doc).get("id")
            print("found something. mate: ", mate, "- score : ", hits[0].score)
            return hits[0], int(mate)
        else:
            return None, -1

        reader.close()


def do_cluster(path, threshold):
    clusters = []
    # repo = DocRepo(path)
    sentences = get_all_sentences()
    random.shuffle(sentences)
    flags = []
    for i in range(0, len(sentences)):
        flags.append(-1)
    repo = DocRepo([])

    print('number of sentences ', len(sentences))
    for senidx, sentence in enumerate(sentences):
        best_macthing_cluster = -1
        # best_macthing_score = -1
        closest, mate = repo.get_most_similar(sentence)
        if (closest is not None) and (closest.score >= threshold):
            #    cluss = [cl.index(mate) for cl in clusters if mate in cl]
            #    if len(cluss) > 0:
            #         best_macthing_cluster = cluss[0]
            #         if len(cluss) > 1:
            #             print("ajibbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
            #             return
            best_macthing_cluster = flags[mate]
        if best_macthing_cluster == -1:
            clusters.append([])
            clusters[-1].append(senidx)
            print(senidx, ' creates new cluster')
            flags[senidx] = len(clusters) - 1
        else:
            print(senidx, ' goes to cluster ', best_macthing_cluster)
            clusters[best_macthing_cluster].append(senidx)
            flags[senidx] = best_macthing_cluster
        repo.addDocument(senidx)
    return clusters


res = do_cluster('IrancellQA.xlsx', threshold=1)

print([len(cluster) for cluster in res])
print(len(res))
cnt = 0
for cl in res:
    if len(cl) == 1:
        print(df2[cols2[1]][cl[0]])
        print("\n")

        cnt += 1
print(cnt)

# analyze = PersianAnalyzer()
# print(analyze.getDefaultStopSet())
# sw  = load_stop_words()
# print(sw)

