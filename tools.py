import random
from record import Record
import pandas as pd
import requests
import json
import os
from statics import email, api_key


def load_stop_words(path=os.path.realpath(os.path.join(__file__, '../persian-stopwords.txt'))):
    stopwords_file = open(path, 'r', encoding='utf8')
    stopwords = stopwords_file.read().split('\n')
    stopwords_file.close()
    return stopwords


def get_clusters(algorithm, corpus, number_of_clusters):
    """
    :param algorithm: a function that get a path and return the clusters
    :param corpus: an array of documents
    :param number_of_clusters: number of clusters that should be make
    :return: a list of Cluster objects
    """
    return algorithm(corpus, number_of_clusters)


def make_corpus(records):
    return [r.a_pre for r in records]


createUrl = "http://api.diaalog.ir/chatbot/intents/create?user=" + email + "&api_key=" + api_key
buildUrl = "http://api.diaalog.ir/chatbot/build?user=" + email + "&api_key=" + api_key


def make_dialog(records, title):
    speechResponse = "{{["
    trainingData = []
    for rec in records:
        speechResponse += '"' + rec.a_pre + '"'
        trainingData.append({"text": rec.q_pre, "entities": []})
        if records[-1] is not rec:
            speechResponse += ','
    speechResponse += "]|random}}"

    print('speechResponse', speechResponse)
    print('trainingData', trainingData)

    data = {'apiTrigger': False,
            'botName': email,
            'intentId': title + '_dialog_' + email,
            'labeledSentences': [],
            'name': title,
            'parameters': [],
            'speechResponse': speechResponse,
            'trainingData': "",
            'userDefined': True
            }

    r = requests.post(url=createUrl, json=data)
    print('create :', r)
    json_response = json.loads(r.content)
    print(json_response)
    id = json_response['_id']
    print('ID is :', id)

    trainAPI = "http://api.diaalog.ir/chatbot/train/" + id + "?user=" + email + "&api_key=" + api_key

    data2 = trainingData
    r2 = requests.post(url=trainAPI, json=data2)
    print('Train :', r2)
    json_response2 = json.loads(r2.content)
    print(json_response2)
    r3 = requests.post(url=buildUrl, json={})

    print('build:', r3)
    json_response3 = json.loads(r3.content)
    print(json_response3)


def build():
    r2 = requests.post(url=buildUrl, json={})
    print('build', r2)


def delete_dialog(id, do_build=True):
    delete = "http://api.diaalog.ir/chatbot/intents/" + id + "?user=" + email + "&api_key=" + api_key
    r = requests.delete(url=delete)
    print('delete :', r)
    if do_build:
        build()


def delete_all_dialogs():
    r = requests.get(url="http://api.diaalog.ir/chatbot/intents?user=" + email + "&api_key=" + api_key)
    print(r)
    json_res = json.loads(r.content)
    id_list = [lis['_id']['$oid'] for lis in json_res]

    for i in range(3, len(id_list)):
        delete_dialog(id_list[i], False)
    build()


def get_answer(clusters, test_string, distance_algorithm, is_lda=False):
    min_dist = 999999999
    best_cluster_index = -1
    for i, c in enumerate(clusters):
        records = c.get_records()
        for r in records:
            dist = distance_algorithm(test_string, r.q_pre)
            if dist < min_dist:
                min_dist = dist
                best_cluster_index = i

    best_cluster_records = clusters[best_cluster_index].get_records()
    if is_lda:
        max_records_index = int(len(best_cluster_records) * .7)
    else:
        max_records_index = len(best_cluster_records) - 1
    return best_cluster_records[random.randint(0, max_records_index)].a_raw


def make_records(df_pre, df_raw):
    records = []
    for each_row in range(df_pre.shape[0]):
        records.append(Record(
            str(df_raw.iat[each_row, 0]),
            str(df_raw.iat[each_row, 1]),
            str(df_pre.iat[each_row, 0]),
            str(df_pre.iat[each_row, 1])))

    return records


def read_data(data_path):
    df_pre = pd.read_excel(data_path, sheet_name='preprocessed')
    df_raw = pd.read_excel(data_path, sheet_name='Raw')
    return df_pre, df_raw


def divide_train_test(records, train_percent):
    random.shuffle(records)
    train_records = records[:int(train_percent * len(records))]
    test_records = records[int(train_percent * len(records)):]
    return train_records, test_records