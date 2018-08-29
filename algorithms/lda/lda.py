from algorithms.lda.lda_gensim import lda_gensim
from algorithms.lda.lda_scikit import lda_scikit


def get_lda(is_gensim=False):
    if is_gensim:
        return lda_gensim
    else:
        return lda_scikit
