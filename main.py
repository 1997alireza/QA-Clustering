import pandas as pd
from algorithms.hierarchical import hierarchical
from algorithms.lda.lda import get_lda
from tools import get_clusters


def read_data(data_path):
    df = pd.read_excel(data_path, sheet_name='preprocessed')
    return df


def make_corpus(pandas_data):
    corpus = []
    for each_row in range(0, pandas_data.shape[0]):
        if type(pandas_data.iat[each_row, 1]) is str:
            corpus.append(pandas_data.iat[each_row, 1])
    return corpus


if __name__ == '__main__':
    data_path = "./QA-samples-reduced.xlsx"
    clusters = get_clusters(hierarchical, make_corpus(read_data(data_path=data_path)))
    top_n_docs = 8
    for x in clusters:
        x.print(top_n_docs)
        print("-------------\n")
