from load_data import get_words_for_each_category
from create_categories import  createCategories2
from nlp import tokenize
from load_embeddings import load_word_vectors
import numpy as np
import os


def create_aspect_emb():
    ''' This a method that creates a vector to initialize the aspect embeddings'''
    # get current directory
    path = os.getcwd()
    # get one directory up
    path = os.path.dirname(path)
    WORD_VECTORS = "../embeddings/glove.6B.300d.txt"
    WORD_VECTORS_DIMS = 300
    print("loading word embeddings...")
    word2idx, idx2word, embeddings = load_word_vectors(WORD_VECTORS,
                                                       WORD_VECTORS_DIMS)

    entity_attribute_pairs = createCategories2()
    TRAIN_DATA = path+"/datasets/ABSA16_Restaurants_Train_SB1_v2.xml"

    d = get_words_for_each_category(entity_attribute_pairs, TRAIN_DATA)

    for i in range(0, 12):
        cat = d[i]
        cat = tokenize(cat)
        cat = set(cat)
        sum_emb = np.zeros(shape=(1, 300))
        word_count = 0
        for word in cat:
            if word in word2idx.keys():
               sum_emb = sum_emb + embeddings[word2idx[word]]
               word_count += 1
        sum_emb = sum_emb/word_count
        with open(path+"/embeddings/aspect_embeddings.txt", "a") as the_file:
            for e in sum_emb:
                for num in e:
                    the_file.write(str(num)+' ')

                the_file.write('\n')
                sum_emb = np.array(sum_emb)
    return sum_emb









