from algorithms.lda.lda import lda

clusters = lda('./QA-samples.xlsx')
top_n_docs = 8
for x in clusters:
    print(x.title, "\n\n\n")
    for i in range(min(len(x.documents()), top_n_docs)):
        print(x.documents[i], '\n')

    print("-------------\n")
