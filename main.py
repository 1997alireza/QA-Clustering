from algorithms.lda.lda import get_lda
from tools import get_clusters

clusters = get_clusters(get_lda(True), './QA-samples.xlsx')
top_n_docs = 8
for x in clusters:
    x.print(top_n_docs)
    print("-------------\n")
