from nltk.tokenize.casual import casual_tokenize
from sklearn.metrics import f1_score
from load_data import read_xml2_train3
from load_embeddings import load_word_vectors
from create_categories import createCategories2
from nlp import vectorize
from sklearn.preprocessing import LabelBinarizer
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
import keras
from keras import Input
from keras.layers import (Embedding, LSTM)
from keras.models import Model as Functional_Model
from keras.optimizers import rmsprop
from keras_models.layers import Attention
from keras.losses import binary_crossentropy
import keras.backend as K
from hyperas.distributions import choice
from keras_models.metrics import f1
import numpy
from keras.layers import GaussianNoise as gausian_noise
from keras.layers import Dropout as dropout
from keras.layers import Dense
import os

#fdfds

def data():
    # get current directory
    path = os.getcwd()
    # get one directory up
    path = os.path.dirname(path)

    WORD_VECTORS = "../embeddings/word2vec.txt"
    WORD_VECTORS_DIMS = 300

    TRAIN_DATA = path+"/datasets/ABSA16_Restaurants_Train_SB1_v2.xml"
    VAL_DATA = path+"/datasets/EN_REST_SB1_TEST.xml"
    max_length = 80
    # load word embeddings
    print("loading word embeddings...")
    word2idx, idx2word, embeddings = load_word_vectors(WORD_VECTORS,WORD_VECTORS_DIMS)
    print("loading categories")
    entity_attribute_pairs = createCategories2()
    # load raw data
    print("loading datasets...")
    train_review, train_ent_attrib = \
        read_xml2_train3(entity_attribute_pairs, TRAIN_DATA)

    gold_review, gold_ent_attrib = \
        read_xml2_train3(entity_attribute_pairs, VAL_DATA)

    y_train = train_ent_attrib
    y_test = gold_ent_attrib
    print("Tokenizing...")
    # nltk tokenizer
    X_train = [casual_tokenize(x, preserve_case=False, reduce_len=True, strip_handles=False) for x in train_review]
    X_test = [casual_tokenize(x, preserve_case=False, reduce_len=True, strip_handles=False) for x in gold_review]
    print("Vectorizing...")
    X_train = numpy.array([vectorize(x, word2idx, max_length) for x in X_train])
    X_test = numpy.array([vectorize(x, word2idx, max_length) for x in X_test])
    print("Turning test and train data to numpy arrays")
    X_train = numpy.array(X_train)
    y_train = numpy.array(y_train)
    X_test = numpy.array(X_test)
    y_test = numpy.array(y_test)
    label_encoder = LabelBinarizer()
    y_train_res = label_encoder.fit_transform(y_train)
    y_test = label_encoder.fit_transform(y_test)
    # Everything to numpy
    X_train = numpy.array(X_train)
    y_train = numpy.array(y_train_res)
    y_test = numpy.array(y_test)
    return embeddings, X_train, X_test, y_train, y_test, max_length


def extract_entities_attributes(embeddings, X_train, X_test, y_train, y_test, max_length):
    # create and return model
    def rnn_entities_attributes5(embeddings, max_length):
        def embedding_layer(embeddings, max_length, trainable=False, masking=False):
            vocab_size = embeddings.shape[0]
            embedding_size = embeddings.shape[1]
            emb_layer = Embedding(
                input_dim=vocab_size,
                output_dim=embedding_size,
                weights=[embeddings],
                input_length=max_length,
                mask_zero=masking,
                trainable=trainable)
            return emb_layer
        ########################################################
        # Optional Parameters
        ########################################################
        rnn_size = 128
        trainable_emb = False
        ########################################################
        inputs = Input(shape=(max_length,), dtype='int32')
        embed = embedding_layer(embeddings=embeddings, max_length=max_length,
                                trainable=trainable_emb)(inputs)
        embed_noise = gausian_noise({{choice([0.2,0.3])}})(embed)
        embed_dropout = dropout({{choice([0.2,0.3,0.4,0.5])}})(embed_noise)
        dense_emb = Dense({{choice([128,256,512,1024])}}, name='dense_after_emb')(embed_dropout)
        lstm_left, state_h_left, state_c_left = LSTM(rnn_size, return_sequences=True, return_state=True)(dense_emb)
        lstm_right, state_h_right, state_c_rigt = LSTM(rnn_size, return_sequences=True, return_state=True,
                                                       go_backwards=True)(dense_emb)
        lstm = keras.layers.concatenate([lstm_left, lstm_right])
        lstm_dropout = dropout({{choice([0.2,0.3,0.4,0.5])}})(lstm)
        atten = Attention()(lstm_dropout)
        output1 = Dense(6, activation='sigmoid')(atten)
        output1_l1 = keras.layers.Lambda(lambda x: K.expand_dims(x, axis=-1))(output1)
        output2 = Dense(5, activation='sigmoid')(atten)
        output2_l2 = keras.layers.Lambda(lambda x: K.expand_dims(x, axis=-1))(output2)
        dot = keras.layers.Dot(axes=2)([output2_l2, output1_l1])
        flatten = keras.layers.Flatten()(dot)
        concat = keras.layers.concatenate([flatten, atten])
        output3 = Dense(256, activation='relu')(concat)
        output3 = Dense(12, activation='sigmoid')(output3)
        model = Functional_Model(inputs=inputs, outputs=output3)
        model.compile(optimizer=rmsprop(), loss=binary_crossentropy, metrics=[f1])
        print(model.summary())
        return model
    model = rnn_entities_attributes5(embeddings=embeddings,max_length=max_length)
    # fit model
    history = model.fit(X_train, y_train,
                        validation_data=(X_test, y_test),
                        epochs=1, batch_size=128)
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))
    preds = model.predict(X_test)
    preds[preds >= 0.5] = 1
    preds[preds < 0.5] = 0
    preds = numpy.array(preds)
    f1_evaluate = f1_score(y_test, preds, pos_label=1, average='micro') * 100
    print("f1 is ",f1_evaluate )
    return {"loss": -f1_evaluate, "status": STATUS_OK, "model": model}

if __name__ == '__main__':
    trials = Trials()
    best_run, best_model = optim.minimize(model=extract_entities_attributes,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=1,
                                          trials=trials,)

    embeddings, X_train, X_test, y_train, y_test, max_length = data()
    print("Evalutation of best performing model:")
    scores = best_model.predict(X_test)
    scores[scores > 0.5] = 1
    scores[scores < 0.5] = 0
    print("Best performing model chosen hyper-parameters:",f1_score(y_test, scores, pos_label=1, average='micro') * 100)
    print(best_run)
    print(trials.miscs)
