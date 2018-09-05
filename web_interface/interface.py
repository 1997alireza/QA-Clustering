from flask import Flask, render_template, request
import requests
from bs4 import BeautifulSoup

from tools import make_records, read_data, divide_train_test, get_clusters, get_answer
from algorithms.hierarchical import hierarchical
# from algorithms.incremental import incremental
from algorithms.lda.lda import get_lda
from similarity_algorithms import euclidean_distance


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/answers')
def answers():
    question = request.args.get('question')
    answer = get_answer(clusters, question, euclidean_distance, is_lda)
    return render_template('answers.html', answer=answer)


def initialize_clustering(data_path, train_percent, clustering_algorithm, number_of_clusters):
    df_pre, df_raw = read_data(data_path=data_path)
    records = make_records(df_pre=df_pre, df_raw=df_raw)
    train_records, test_records = divide_train_test(records=records, train_percent=train_percent)
    return get_clusters(clustering_algorithm, train_records, number_of_clusters)


if __name__ == '__main__':
    data_path = "../QA-samples-reduced.xlsx"
    train_percent = 0.8
    number_of_clusters = 900
    clustering_algorithm = get_lda(False)

    clusters = initialize_clustering(data_path, train_percent, clustering_algorithm, number_of_clusters)
    is_lda = clustering_algorithm == get_lda(True) or clustering_algorithm == get_lda(False)

    print("clusters are ready")

    app.run()
