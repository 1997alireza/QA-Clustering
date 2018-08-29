import random

import numpy as np
import matplotlib.pyplot as plt
from nltk.metrics import edit_distance


# preform test and plot record
def preform_test(clusters, test_records, distance_algorithm, clustering_algorithm_name, distance_algorithm_name):
    distances = []
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

    print("Mean = ", np.mean(distances), " Variance = ", np.var(distances), "\n", distances)
    plt.hist(distances, bins=range(0, 50))
    plt.title("Histogram of " + clustering_algorithm_name + " algorithm with " + distance_algorithm_name,)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()

    # print(test_records[i].q_pre, "      ", "       ", recommended_answer, "\n",
    #       "***************************")


def draw_histogram():
    print("")
