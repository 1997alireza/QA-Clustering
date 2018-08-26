from algorithms.lda.lda_gensim import lda_gensim
from algorithms.lda.lda_scikit import lda_scikit


def do_lda(path, is_gensim=False):
    if is_gensim:
        return lda_gensim(path)
    else:
        return lda_scikit(path)
