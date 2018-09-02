def load_stop_words():
    stopwords_file = open('persian-stopwords.txt', 'r', encoding='utf8')
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


import requests
import json
from record import Record

api_key = "eJGlFIowIM"
createUrl = "http://api.diaalog.ir/chatbot/intents/create?user=torabian.alireza@gmail.com&api_key=" + api_key
buildUrl = "http://api.diaalog.ir/chatbot/build?user=torabian.alireza@gmail.com&api_key=" + api_key


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
            'botName': 'torabian.alireza@gmail.com',
            'intentId': title + '_dialog_torabian.alireza@gmail.com',
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

    trainAPI = "http://api.diaalog.ir/chatbot/train/" + id + "?user=torabian.alireza@gmail.com&api_key=" + api_key

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
    delete = "http://api.diaalog.ir/chatbot/intents/" + id + "?user=torabian.alireza@gmail.com&api_key=" + api_key
    r = requests.delete(url=delete)
    print('delete :', r)
    if do_build:
        build()


def delete_all_dialogs():
    r = requests.get(url="http://api.diaalog.ir/chatbot/intents?user=torabian.alireza@gmail.com&api_key=" + api_key)
    print(r)
    json_res = json.loads(r.content)
    id_list = [lis['_id']['$oid'] for lis in json_res]

    for i in range(3, len(id_list)):
        delete_dialog(id_list[i], False)
    build()


delete_all_dialogs()
