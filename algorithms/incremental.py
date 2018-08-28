import lucene

from org.apache.lucene.analysis.standard import StandardAnalyzer, StandardTokenizer
from org.apache.lucene.store import RAMDirectory
from org.apache.lucene.index import IndexWriter
from org.apache.lucene.index import IndexWriterConfig
from org.apache.lucene.document import Field
from org.apache.lucene.document import StringField
from org.apache.lucene.document import Document
import pandas as pd
from org.apache.lucene.search import IndexSearcher, BooleanClause, BooleanQuery, TermQuery
from org.apache.lucene.index import DirectoryReader, Term

lucene.initVM(vmargs=['-Djava.awt.headless=true'])

data = pd.ExcelFile('../QA-samples-reduced.xlsx')
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
        rawA = df2[cols2[0]][i]
        sentences.append(rawA)
    return sentences


class DocRepo:

    def __init__(self, indexes=None):

        self.analyzer = StandardAnalyzer()
        self.config = IndexWriterConfig(self.analyzer)
        self.index = RAMDirectory()
        if indexes is not None:
            self.loadIndex(indexes)
        else:
            self.loadIndex()
        # print("hiiiiiii")

    def loadIndex(self, indexes=None):
        # print("loading")
        self.w = IndexWriter(self.index, self.config)
        if indexes is None:
            for i in range(0, 14191):
                rawQ = df1[cols1[0]][i]
                rawA = df2[cols2[0]][i]
                preQ = df1[cols1[1]][i]
                self.addDoc(i, rawQ, preQ, rawA)
        else:
            for i in indexes:
                rawQ = df1[cols1[0]][i]
                rawA = df2[cols2[0]][i]
                preQ = df1[cols1[1]][i]
                self.addDoc(i, rawQ, preQ, rawA)

    def addDoc(self, id, rawQuestion, preprocessedQuestion, RawAnswer):
        doc = Document()
        doc.add(StringField("rq", rawQuestion, Field.Store.YES))
        doc.add(StringField("pq", preprocessedQuestion, Field.Store.YES))
        doc.add(StringField("ra", RawAnswer, Field.Store.YES))
        doc.add(StringField("id", str(id), Field.Store.YES))
        self.w.addDocument(doc)

    def addDocument(self, id):
        rawQ = df1[cols1[0]][id]
        rawA = df2[cols2[0]][id]
        preQ = df1[cols1[1]][id]
        doc = Document()
        doc.add(StringField("rq", rawQ, Field.Store.YES))
        doc.add(StringField("pq", preQ, Field.Store.YES))
        doc.add(StringField("ra", rawA, Field.Store.YES))
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
            qtq = TermQuery(Term("ra", q_word))
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
    repo = DocRepo([])

    print('number of sentences ', len(sentences))
    for senidx, sentence in enumerate(sentences):
        best_macthing_cluster = -1
        # best_macthing_score = -1
        closest, mate = repo.get_most_similar(sentence)
        if (closest is not None) and (closest.score >= threshold):
            cluss = [cl.index(mate) for cl in clusters if mate in cl]
            if len(cluss) > 0:
                best_macthing_cluster = cluss[0]
                # if len(cluss) > 1:
                #     print("ajibbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
                #     return
        if best_macthing_cluster == -1:
            clusters.append([senidx])
            print(senidx, ' creates new cluster')
        else:
            print(senidx, ' goes to cluster ', best_macthing_cluster)
            clusters[best_macthing_cluster].append(senidx)
        repo.addDocument(senidx)
    return clusters


clusters = do_cluster('../QA-samples-reduced.xlsx', threshold=1)

print([len(cluster) for cluster in clusters])
print(len(clusters))