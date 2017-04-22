import gensim
import os
import collections
import smart_open
import random
import logging
import time

""" Reference:

    Doc2Vec Tutorial on the Lee Dataset:
    https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/doc2vec-lee.ipynb

    gensim doc2vec & IMDB sentiment dataset:
    https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/doc2vec-IMDB.ipynb

"""

corpus_dir = ""
TRAIN_DOC_NUM = 5000

print(corpus_dir)

def get_doc_list():
    """Get all the log file name."""
    doc_list = []
    for fname in doc_list:#os.listdir(corpus_dir):
        doc_list.append(fname)
    return shuffle(doc_list)

def read_corpus(doc_list, tokens_only=False):
    """Tag all docs, one doc coresspands to one TaggedDocument object."""
    for index, fname in enumerate(doc_list):
        file = corpus_dir + "/" + fname
        with smart_open.smart_open(file, encoding="iso-8859-1") as f:
            content = f.read()
            if(tokens_only):
                yield gensim.utils.simple_preprocess(content)
            else:
                # For test data, add tags
                yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(content), [fname])


def evaluate(train_corpus, model):
    ranks = []
    second_ranks = []
    for doc_id in range(len(train_corpus)):
        inferred_vector = model.infer_vector(train_corpus[doc_id].words)
        sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
        rank = [docid for docid, sim in sims].index(doc_id)
        ranks.append(rank)
        
        second_ranks.append(sims[1])
    return collections.Counter(ranks)

def main():
    doc_list = get_doc_list()
    train_corpus = list(read_corpus(random.sample(doc_list, TRAIN_DOC_NUM)))
    test_corpus = list(read_corpus(random.sample(doc_list, len(log_lsit) - TRAIN_DOC_NUM), true)

    print("Get train corpus generator successfully.")

    '''
        dm defines the training algorithm, defaults to PV-DM, otherwise PV-DBOW.
        size is the dimensionality of the feature vectors.
        window is the maximum distance between the predicted word and context words used for prediction within a document.
        alpha is the initial learning rate (will linearly drop to zero as training progresses).
        min_count = ignore all words with total frequency lower than this.
        iter = number of iterations (epochs) over the corpus.
    '''
    model = gensim.models.doc2vec.Doc2Vec(dm = 0, alpha = 0.025, size = 3000, min_alpha = 0.025, min_count = 0, workers = 4, iter = 20)
    model.build_vocab(train_corpus)

    model.train(train_corpus)

    curTime = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    savedModelName = "./models/trained_8050_logs_" + curTime + ".model"
    model.save(savedModelName)
    docvecs = model.docvecs

    print evaluate(train_corpus, model)

if __name__ == "__main__":
    main()
