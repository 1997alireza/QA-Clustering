from algorithms.hierarchical import hierarchical
from algorithms.incremental import incremental
from algorithms.lda.lda import get_lda
from clustering_test import preform_test
from similarity_algorithms import euclidean_distance
from tools import get_clusters, make_records, read_data, divide_train_test

if __name__ == '__main__':
    data_path = "QA-samples.xlsx"
    train_percent = 0.8
    number_of_clusters = 900
    clustering_algorithm = get_lda(False)

    df_pre, df_raw = read_data(data_path=data_path)
    records = make_records(df_pre=df_pre, df_raw=df_raw)[:100]
    train_records, test_records = divide_train_test(records=records, train_percent=train_percent)
    clusters = get_clusters(clustering_algorithm, train_records, number_of_clusters)
    is_lda = clustering_algorithm == get_lda(True) or clustering_algorithm == get_lda(False)
    preform_test(clusters, test_records, euclidean_distance, clustering_algorithm_name=clustering_algorithm.__name__,
                 is_lda=is_lda, number_of_clusters=number_of_clusters)
