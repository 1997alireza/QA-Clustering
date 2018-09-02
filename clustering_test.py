import random

import numpy as np
import matplotlib.pyplot as plt
from nltk.metrics import edit_distance


# preform test and plot record
def preform_test(clusters, test_records, distance_algorithm, clustering_algorithm_name, is_lda=False,number_of_clusters =0):
    distances = []
    for i in range(len(test_records)):
        min_questions_dist = 10000000
        min_cluster_index = 1000000
        for j in range(len(clusters)):
            records_cj = clusters[j].get_records()
            for k in range(len(records_cj)):
                tmp = distance_algorithm(test_records[i].q_pre, records_cj[k].q_pre)
                if tmp < min_questions_dist:
                    min_questions_dist = tmp
                    min_cluster_index = j

        best_cluster_recrods = clusters[min_cluster_index].get_records()
        if is_lda:
            max_records_index = int(len(best_cluster_recrods) * .7)
        else:
            max_records_index = len(best_cluster_recrods) - 1
        recommended_answer = best_cluster_recrods[
            random.randint(0, max_records_index)].a_pre
        distances.append(distance_algorithm(recommended_answer, test_records[i].a_pre))

    print("Mean = ", np.mean(distances), " Variance = ", np.var(distances), "\n")
    plt.hist(distances, bins=range(0, 60))
    plt.title("Histogram of " + clustering_algorithm_name + " algorithm with " + distance_algorithm.__name__ + str(number_of_clusters))
    plt.xlabel("Difference")
    plt.ylabel("Frequency")
    plt.show()

    # print(test_records[i].q_pre, "      ", "       ", recommended_answer, "\n",
    #       "***************************")


def draw_histogram():
    print("")
