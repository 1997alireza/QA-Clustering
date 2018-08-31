import random

import pandas as pd
from algorithms.hierarchical import hierarchical
from algorithms.incremental import incremental
from algorithms.lda.lda import get_lda
from record import Record
from tools import get_clusters
from clustering_test import preform_test
from similarity_algorithms import euclidean_distance


def read_data(data_path):
    df_pre = pd.read_excel(data_path, sheet_name='preprocessed')
    df_raw = pd.read_excel(data_path, sheet_name='Raw')
    return df_pre, df_raw


def make_records(df_pre, df_raw):
    records = []
    for each_row in range(df_pre.shape[0]):
        records.append(Record(
            str(df_raw.iat[each_row, 0]),
            str(df_raw.iat[each_row, 1]),
            str(df_pre.iat[each_row, 0]),
            str(df_pre.iat[each_row, 1])))

    return records[:1500]


def divide_train_test(records, train_percent):
    random.shuffle(records)
    train_records = records[:int(train_percent * len(records))]
    test_records = records[int(train_percent * len(records)):]
    return train_records, test_records


if __name__ == '__main__':
    data_path = "QA-samples.xlsx"
    train_percent = 0.9
    number_of_clusters = 150
    df_pre, df_raw = read_data(data_path=data_path)
    records = make_records(df_pre=df_pre, df_raw=df_raw)
    train_records, test_records = divide_train_test(records=records, train_percent=train_percent)
    clustering_algorithm = get_lda(False)
    is_lda = clustering_algorithm == get_lda(True) or clustering_algorithm == get_lda(False)
    clusters = get_clusters(clustering_algorithm, train_records, number_of_clusters)
    preform_test(clusters, test_records, euclidean_distance, clustering_algorithm_name=clustering_algorithm.__name__,
                 is_lda=is_lda)
    top_n_docs = 8
    # for x in clusters:
    #     x.print(top_n_docs, True)
    #     print("-------------\n")
