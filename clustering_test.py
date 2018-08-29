import random

from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
from nltk.metrics import edit_distance


def preform_test(clusters, train_records, test_records):
    distances = []
    distance_algorithm = edit_distance
    for i in range(len(test_records)):
        min_questions_dist = 10000000
        min_cluster_index = 1000000
        for j in range(len(clusters)):
            for k in range(len(clusters[j].records)):
                tmp = distance_algorithm(test_records[i].q_pre, clusters[j].records[k].q_pre)
                if tmp < min_questions_dist:
                    min_questions_dist = tmp
                    min_cluster_index = j

        recommended_answer = clusters[min_cluster_index].records[
            random.randint(0, len(clusters[min_cluster_index].records) - 1)].a_pre
        distances.append(distance_algorithm(recommended_answer, test_records[i].a_pre))

    print(distances)
    plt.hist(distances, bins=range(0, 600))
    plt.title("Histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()

    # print(test_records[i].q_pre, "      ", "       ", recommended_answer, "\n",
    #       "***************************")


def draw_histogram():
    print("")
