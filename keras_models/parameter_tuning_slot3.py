from __future__ import print_function
import numpy
from nlp import tokenize, extract_setiment_scores
from load_data import read_xml_polarities
from load_embeddings import load_word_vectors
import keras
from keras import Input
from keras.layers import (Embedding, LSTM,)
from keras.models import Model as Functional_Model
from keras_models.layers import Attention
from  keras.losses import categorical_crossentropy
from keras_models.metrics import accuracy2
from hyperopt import Trials, STATUS_OK, tpe
from keras.layers.core import Dense
from hyperas import optim
from hyperas.distributions import choice, uniform
from create_categories import createCategories2, create_polarities
from nlp import vectorize
import os

def data():
    # get current directory
    path = os.getcwd()
    # get one directory up
    path = os.path.dirname(path)

    WORD_VECTORS = "../embeddings/glove.6B.300d.txt"
    TRAIN_DATA = path+"/datasets/ABSA16_Restaurants_Train_SB1_v2.xml"
    VAL_DATA = path+"/ABSA/datasets/EN_REST_SB1_TEST.xml"
    BATCH_SIZE = 128
    EPOCHS = 50
    WORD_VECTORS_DIMS = 300
    MAX_LENGTH = 80
    max_length = 80
    _hparams = {
        "rnn_size": 100,
        "bidirectional": True,
        "noise": 0.2,
        "dropout_words": 0.2,
        "dropout_rnn": 0.5,
    }
    # load word embeddings
    print("loading word embeddings...")
    word2idx, idx2word, embeddings = load_word_vectors(WORD_VECTORS,WORD_VECTORS_DIMS)
    print("loading categories")
    entity_attribute_pairs = createCategories2()
    polarity_labels = create_polarities()
    # load raw data
    print("loading datasets...")
    train_review, train_ent_attrib, train_polarity,train_aux = \
        read_xml_polarities(entity_attribute_pairs,polarity_labels,TRAIN_DATA)
    gold_review, gold_ent_attrib, gold_polarity,gold_aux = \
        read_xml_polarities(entity_attribute_pairs,polarity_labels,VAL_DATA)
    print("extracting sentiment from texts")
    sentiment_intensity_train = extract_setiment_scores(train_review)
    sentiment_intensity_test = extract_setiment_scores(gold_review)
    y_train = train_polarity
    y_test = gold_polarity
    words = gold_review
    print("Tokenizing...")
    X_train = [tokenize(x) for x in train_review]
    X_test = [tokenize(x) for x in gold_review]
    print("Vectorizing...")
    X_train = numpy.array([vectorize(x, word2idx, MAX_LENGTH) for x in X_train])
    X_test = numpy.array([vectorize(x, word2idx, MAX_LENGTH) for x in X_test])
    print("Turning test and train data to numpy arrays")
    # train and test sentence
    X_train = numpy.array(X_train)
    X_test = numpy.array(X_test)
    # train and test E#A labels
    y_train = numpy.array(y_train)
    y_test = numpy.array(y_test)
    # auxilary input - the aspect that is present in the sentence
    train_aux = numpy.array(train_aux)
    gold_aux = numpy.array(gold_aux)
    # handcrafted feature - the sentiment intensity of each sentence in a 3 way scale
    sentiment_intensity_train = numpy.array(sentiment_intensity_train)
    sentiment_intensity_test = numpy.array(sentiment_intensity_test)
    # entity and attribute pairs are given in this task and we can use them as extra features in each sentence
    train_ent_attrib = numpy.array(train_ent_attrib)
    gold_ent_attrib = numpy.array(gold_ent_attrib)
    classes = len(polarity_labels)
    return embeddings, classes, max_length, X_train, train_aux, sentiment_intensity_train, y_train, X_test, gold_aux, sentiment_intensity_test, y_test


def aspect_polarity(embeddings, classes, max_length, X_train, train_aux, sentiment_intensity_train, y_train, X_test, gold_aux, sentiment_intensity_test, y_test):
    rnn_size = 128
    max_length= 80

    vocab_size = embeddings.shape[0]
    embedding_size = embeddings.shape[1]
    emb_layer = Embedding(input_dim=vocab_size,output_dim=embedding_size,
                          weights=[embeddings],input_length=max_length, mask_zero=False, trainable=False, name='word_embedding')
    vocab_size = 13
    embedding_size = embeddings.shape[1]
    emb_layer2 = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_size,
        input_length=1,
        mask_zero=False,
        trainable=True,name='aspect_embedding')
    sent_input = Input(shape=(max_length,), dtype='int32', name='sentence')
    embed = emb_layer(sent_input)
    intensity_input = Input(shape=(3,),dtype='float32', name='setiment_intesity')
    auxilary_input = Input(shape=(1, ), dtype='int32', name='auxilary_input')
    aspect_embed = emb_layer2(auxilary_input)
    aspect_embed = keras.layers.Flatten(name='auxilary_flattened')(aspect_embed)
    aspect_embed = keras.layers.RepeatVector(max_length, name='auxilary_repeated')(aspect_embed)
    concat = keras.layers.concatenate([embed, aspect_embed], name='concatenated_embeddings')
    concat = keras.layers.GaussianNoise({{uniform(0, 1)}}, name='gausian_noise')(concat)
    concat = keras.layers.Dropout({{uniform(0, 1)}}, name='dropout_concat_emb')(concat)
    lstm, state_h, state_c = LSTM({{choice([64, 128])}}, return_sequences=True, return_state=True, name='lstm_layer')(concat)
    lstm = keras.layers.Dropout({{uniform(0, 1)}})(lstm)
    hidden = keras.layers.concatenate([lstm, aspect_embed], name='concatenated_lstm_embedding')
    attention = Attention(name='attention')(hidden)
    state_c = keras.layers.concatenate([state_c,intensity_input])
    atten_final_state = keras.layers.concatenate([attention,state_c])
    output = Dense(units={{choice([50, 150,250])}}, activation={{choice(['relu','tanh'])}}, name='dense_relu')(atten_final_state)
    output = Dense(classes, activation='softmax', name='dense_softmax')(output)
    model = Functional_Model(inputs=[sent_input,auxilary_input,intensity_input], outputs=output)
    model.compile(optimizer={{choice(['adam','rmsprop'])}}, loss=categorical_crossentropy, metrics=['accuracy'])
    print(model.summary())
    model.fit([X_train, train_aux, sentiment_intensity_train],
              y_train, epochs=1, batch_size={{choice([64, 128])}},
              validation_data=([X_test, gold_aux, sentiment_intensity_test], y_test))
    scores = model.predict([X_test, gold_aux, sentiment_intensity_test])
    threshhold = {{uniform(0, 1)}}
    scores[scores > threshhold] = 1
    scores[scores < (1-threshhold)] = 0
    acc = accuracy2(y_test, scores)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=aspect_polarity,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=1,
                                          trials=Trials())
    embeddings, classes, max_length, X_train, train_aux, sentiment_intensity_train, y_train, X_test, gold_aux, sentiment_intensity_test, y_test= data()
    print("Evalutation of best performing model:")
    scores = best_model.predict([X_test, gold_aux, sentiment_intensity_test])
    scores[scores > 0.5] = 1
    scores[scores < 0.5] = 0
    print(accuracy2(y_test,scores))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
