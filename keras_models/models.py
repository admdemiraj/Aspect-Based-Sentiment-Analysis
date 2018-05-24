import keras
from keras import Input
from keras.layers import (Embedding,  LSTM,
                          Dense,  Bidirectional)
from keras.models import Model as Functional_Model
from keras.optimizers import Adam, rmsprop

from keras_models.layers import Attention
from keras.losses import categorical_crossentropy, binary_crossentropy
from keras_models.metrics import f1


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


def embedding_layer2( trainable=True, masking=False):
    vocab_size = 13
    embedding_size = 50
    emb_layer = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_size,
        input_length=1,
        mask_zero=masking,
        trainable=trainable)
    return emb_layer


# ENTITY ATTRIBUTE MODELS#

def entities_attributes_attention_dense(embeddings, max_length):

    ''' BASELINE #1
     This a simple baseline model using an embedding layer ,
     afterwards an attention layer and on top a simple dense
     layer with a sigmoid to take a decision for all the E#A pairs '''

    inputs = Input(shape=(max_length, ), dtype='int32')
    embed = embedding_layer(embeddings=embeddings, max_length=max_length, trainable=False, masking=False)(inputs)
    atten = Attention()(embed)
    output = Dense(13, activation='sigmoid')(atten)
    model = Functional_Model(inputs=inputs, outputs=output)
    model.compile(optimizer=rmsprop(), loss=binary_crossentropy, metrics=[f1])
    return model


def entities_attributes_LSTM_attention_dense(embeddings, max_length):

    ''' BASELINE #2
     This a simple baseline model using an embedding layer ,
     afterwards a LSTM with an attention layer and on top a simple dense
     layer with a sigmoid to take a decision for all the E#A pairs '''

    inputs = Input(shape=(max_length, ), dtype='int32')
    embed = embedding_layer(embeddings=embeddings, max_length=max_length, trainable=False, masking=False)(inputs)
    lstm = LSTM(units=128,return_sequences=True)(embed)
    atten = Attention()(lstm)
    output = Dense(13, activation='sigmoid')(atten)
    model = Functional_Model(inputs=inputs, outputs=output)
    model.compile(optimizer=rmsprop(), loss=binary_crossentropy, metrics=[f1])
    return model


def entities_attributes_LSTM_attention_dence_dropout(embeddings, max_length):
    ''' BASELINE 3
    This is a model that uses embedding layer,a dropout layer, afterwards a LSTM layer with a
    self attention mechanism on top, a dropout layer and finally a dense layer
    to make the decision'''

    inputs = Input(shape=(max_length,), dtype='int32')
    embed = embedding_layer(embeddings=embeddings, max_length=max_length, trainable=False, masking=False)(inputs)
    embed_noise = keras.layers.GaussianNoise(0.2)(embed)
    embed_dropout = keras.layers.Dropout(0.3, noise_shape=(1, 300))(embed_noise)
    lstm = LSTM(units=128, return_sequences=True)(embed_dropout)
    atten = Attention()(lstm)
    atten_dropout = keras.layers.Dropout(0.2, noise_shape=(1, 128))(atten)
    output = Dense(13, activation='sigmoid')(atten_dropout)
    model = Functional_Model(inputs=inputs, outputs=output)
    model.compile(optimizer=rmsprop(), loss=binary_crossentropy, metrics=[f1])
    return model


def entities_attributes_projection_LSTM_attention_dence_dropout(embeddings, max_length):
    ''' BASELINE 4
    This is a model that uses embedding layer, a projection of the embeddings , a dropout layer,
     afterwards a LSTM layer with a self attention mechanism on top, a dropout layer and finally
     a dense layer to make the decision'''

    inputs = Input(shape=(max_length,), dtype='int32')
    embed = embedding_layer(embeddings=embeddings, max_length=max_length, trainable=False, masking=False)(inputs)
    embed_noise = keras.layers.GaussianNoise(0.2)(embed)
    embed_dropout = keras.layers.Dropout(0.3, noise_shape=(1, 300))(embed_noise)
    dense_emb = keras.layers.Dense(1024, activation='tanh', name='dense_after_emb')(embed_dropout)
    lstm = LSTM(units=128, return_sequences=True)(dense_emb)
    atten = Attention()(lstm)
    atten_dropout = keras.layers.Dropout(0.2, noise_shape=(1, 128))(atten)
    output = Dense(13, activation='sigmoid')(atten_dropout)
    model = Functional_Model(inputs=inputs, outputs=output)
    model.compile(optimizer=rmsprop(), loss=binary_crossentropy, metrics=[f1])
    return model


def entities_attributes_projection_LSTM_attention_dencex2_dropout(embeddings, max_length):
    ''' BASELINE 5
    This is a model that uses embedding layer, a projection of the embeddings , a dropout layer,
     afterwards a LSTM layer with a self attention mechanism on top, a dropout layer and finally
     a  2 dense layers (one with activation relu) to make the decision'''

    inputs = Input(shape=(max_length,), dtype='int32')
    embed = embedding_layer(embeddings=embeddings, max_length=max_length, trainable=False, masking=False)(inputs)
    embed_noise = keras.layers.GaussianNoise(0.2)(embed)
    embed_dropout = keras.layers.Dropout(0.3, noise_shape=(1, 300))(embed_noise)
    dense_emb = keras.layers.Dense(1024, activation='tanh', name='dense_after_emb')(embed_dropout)
    lstm = LSTM(units=128, return_sequences=True)(dense_emb)
    atten = Attention()(lstm)
    atten_dropout = keras.layers.Dropout(0.2, noise_shape=(1, 128))(atten)
    output = Dense(250, activation='relu')(atten_dropout)
    output = Dense(13, activation='sigmoid')(output)
    model = Functional_Model(inputs=inputs, outputs=output)
    model.compile(optimizer=rmsprop(), loss=binary_crossentropy, metrics=[f1])
    return model

def entities_attributes_projection_BiLSTM_attention_dencex2_dropout(embeddings, max_length):
    ''' BASELINE 6
    This is a model that uses embedding layer, a projection of the embeddings , a dropout layer,
     afterwards a Bi_LSTM layer with a self attention mechanism on top, a dropout layer and finally
     a  2 dense layers (one with activation relu) to make the decision'''

    inputs = Input(shape=(max_length,), dtype='int32')
    embed = embedding_layer(embeddings=embeddings, max_length=max_length, trainable=False, masking=False)(inputs)
    embed_noise = keras.layers.GaussianNoise(0.2)(embed)
    embed_dropout = keras.layers.Dropout(0.3, noise_shape=(1, 300))(embed_noise)
    dense_emb = keras.layers.Dense(1024, activation='tanh', name='dense_after_emb')(embed_dropout)
    lstm = keras.layers.Bidirectional(LSTM(units=128, return_sequences=True))(dense_emb)
    atten = Attention()(lstm)
    atten_dropout = keras.layers.Dropout(0.2, noise_shape=(1, 256))(atten)
    output = Dense(250, activation='relu')(atten_dropout)
    output = Dense(13, activation='sigmoid')(output)
    model = Functional_Model(inputs=inputs, outputs=output)
    model.compile(optimizer=rmsprop(), loss=binary_crossentropy, metrics=[f1])
    return model


# POLARITY MODELS#


def embeddings_attention(embeddings, classes, max_length):
    ''' BASELINE 1
    A simple model having an embedding layer with attention on top and a softmax for the final output'''

    # create sent embeddings
    sent_input = Input(shape=(max_length,), dtype='int32', name='sentence')
    embed = embedding_layer(embeddings=embeddings, max_length=max_length
                            , trainable=False)(sent_input)
    attention = Attention(name='attention')(embed)
    output = Dense(classes, activation='softmax', name='dense_softmax')(attention)
    model = Functional_Model(inputs=sent_input, outputs=output)
    model.compile(optimizer=Adam(), loss=categorical_crossentropy, metrics=['accuracy'])
    print(model.summary())
    return model


def emd_conc_asp_emb_attention(embeddings, classes, max_length):
    ''' BASELINE 2
     A simple model having a concatenation of word and aspect embeddings with
     attention on top and a softmax for the final output'''

    # create sent embeddings
    sent_input = Input(shape=(max_length,), dtype='int32', name='sentence')
    embed = embedding_layer(embeddings=embeddings, max_length=max_length
                            , trainable=False)(sent_input)
    # create aspect embeddings
    auxilary_input = Input(shape=(1, ), dtype='int32', name='auxilary_input')
    aspect_embed = embedding_layer2(embeddings=embeddings, trainable=True)(auxilary_input)
    # bring the aspect embeddings in the appropriate format (we need to repeat the same aspect embedding k times)
    aspect_embed = keras.layers.Flatten(name='auxilary_flattened')(aspect_embed)
    aspect_embed = keras.layers.RepeatVector(max_length, name='auxilary_repeated')(aspect_embed)
    # concatenate the aspect embedding with the sent embedding
    concat = keras.layers.concatenate([embed, aspect_embed], name='concatenated_embeddings')
    attention = Attention(name='attention')(concat)
    output = Dense(classes, activation='softmax', name='dense_softmax')(attention)
    model = Functional_Model(inputs=[sent_input, auxilary_input], outputs=output)
    model.compile(optimizer=rmsprop(), loss=categorical_crossentropy, metrics=['accuracy'])
    print(model.summary())
    return model


def emd_conc_asp_emb_LSTM_attention(embeddings, classes, max_length):
    ''' BASELINE 3
     A simple model having a concatenation of word and aspect embeddings,a LSTM
     cell after that and attention on top and a softmax for the final output'''

    rnn_size = 128
    # create sent embeddings
    sent_input = Input(shape=(max_length,), dtype='int32', name='sentence')
    embed = embedding_layer(embeddings=embeddings, max_length=max_length
                            , trainable=False)(sent_input)
    # create aspect embeddings
    auxilary_input = Input(shape=(1, ), dtype='int32', name='auxilary_input')
    aspect_embed = embedding_layer2(embeddings=embeddings, trainable=True)(auxilary_input)
    # bring the aspect embeddings in the appropriate format (we need to repeat the same aspect embedding k times)
    aspect_embed = keras.layers.Flatten(name='auxilary_flattened')(aspect_embed)
    aspect_embed = keras.layers.RepeatVector(max_length, name='auxilary_repeated')(aspect_embed)
    # concatenate the aspect embedding with the sent embedding
    concat = keras.layers.concatenate([embed, aspect_embed], name='concatenated_embeddings')
    lstm = LSTM(rnn_size, return_sequences=True)(concat)
    attention = Attention(name='attention')(lstm)
    output = Dense(classes, activation='softmax', name='dense_softmax')(attention)
    model = Functional_Model(inputs=[sent_input, auxilary_input], outputs=output)
    model.compile(optimizer=rmsprop(), loss=categorical_crossentropy, metrics=['accuracy'])
    print(model.summary())
    return model


def emd_conc_asp_emb_dropout_LSTM_attention(embeddings, classes, max_length):
    ''' BASELINE 4
     A simple model having a concatenation of word and aspect embeddings with gaussian
     noise,a dropout layer,a LSTM cell, attention on top,a final dropout layer and a
     softmax for the final output'''

    rnn_size = 128
    # create sent embeddings
    sent_input = Input(shape=(max_length,), dtype='int32', name='sentence')
    embed = embedding_layer(embeddings=embeddings, max_length=max_length
                            , trainable=False)(sent_input)
    # create aspect embeddings
    auxilary_input = Input(shape=(1,), dtype='int32', name='auxilary_input')
    aspect_embed = embedding_layer2(embeddings=embeddings, trainable=True)(auxilary_input)
    # bring the aspect embeddings in the appropriate format (we need to repeat the same aspect embedding k times)
    aspect_embed = keras.layers.Flatten(name='auxilary_flattened')(aspect_embed)
    aspect_embed = keras.layers.RepeatVector(max_length, name='auxilary_repeated')(aspect_embed)
    # concatenate the aspect embedding with the sent embedding
    concat = keras.layers.concatenate([embed, aspect_embed], name='concatenated_embeddings')
    concat = keras.layers.GaussianNoise(0.2)(concat)
    concat = keras.layers.Dropout(0.2, noise_shape=(1, 350), name='dropout_concat_emb', )(concat)
    lstm = LSTM(rnn_size, return_sequences=True)(concat)
    attention = Attention(name='attention')(lstm)
    atten_dropout = keras.layers.Dropout(0.2, noise_shape=(1, 128))(attention)
    output = Dense(classes, activation='softmax', name='dense_softmax')(atten_dropout)
    model = Functional_Model(inputs=[sent_input, auxilary_input], outputs=output)
    model.compile(optimizer=rmsprop(), loss=categorical_crossentropy, metrics=['accuracy'])
    print(model.summary())
    return model

def conc_emb_dropout_LSTM_conc_emb_attention(embeddings, classes, max_length):
    ''' BASELINE 5
     A simple model having a concatenation of word and aspect embeddings with gaussian
     noise,a dropout layer,a LSTM cell,a concatenation of each hidden state with an aspect
     embedding, attention on top,a final dropout layer and a softmax for the final output'''

    rnn_size = 128
    # create sent embeddings
    sent_input = Input(shape=(max_length,), dtype='int32', name='sentence')
    embed = embedding_layer(embeddings=embeddings, max_length=max_length
                            , trainable=False)(sent_input)
    # create aspect embeddings
    auxilary_input = Input(shape=(1,), dtype='int32', name='auxilary_input')
    aspect_embed = embedding_layer2(embeddings=embeddings, trainable=True)(auxilary_input)
    # bring the aspect embeddings in the appropriate format (we need to repeat the same aspect embedding k times)
    aspect_embed = keras.layers.Flatten(name='auxilary_flattened')(aspect_embed)
    aspect_embed = keras.layers.RepeatVector(max_length, name='auxilary_repeated')(aspect_embed)
    # concatenate the aspect embedding with the sent embedding
    concat = keras.layers.concatenate([embed, aspect_embed], name='concatenated_embeddings')
    concat = keras.layers.GaussianNoise(0.2)(concat)
    concat = keras.layers.Dropout(0.2, noise_shape=(1, 350), name='dropout_concat_emb', )(concat)
    lstm = LSTM(rnn_size, return_sequences=True)(concat)
    hidden = keras.layers.concatenate([lstm, aspect_embed], name='concatenated_lstm_embedding')
    attention = Attention(name='attention')(hidden)
    atten_dropout = keras.layers.Dropout(0.2, noise_shape=(1, 178))(attention)
    output = Dense(classes, activation='softmax', name='dense_softmax')(atten_dropout)
    model = Functional_Model(inputs=[sent_input, auxilary_input], outputs=output)
    model.compile(optimizer=rmsprop(), loss=categorical_crossentropy, metrics=['accuracy'])
    print(model.summary())
    return model


def conc_emb_projection_dropout_LSTM_conc_emb_attention(embeddings, classes, max_length):
    ''' BASELINE 6
     A simple model having a concatenation of word and aspect embeddings with gaussian
     noise,a projection layer with a tanh activation, a dropout layer,a LSTM cell,
     a concatenation of each hidden state with an aspect embedding, attention on top,
     a final dropout layer and a softmax for the final output'''

    rnn_size = 128
    # create sent embeddings
    sent_input = Input(shape=(max_length,), dtype='int32', name='sentence')
    embed = embedding_layer(embeddings=embeddings, max_length=max_length
                            , trainable=False)(sent_input)
    # create aspect embeddings
    auxilary_input = Input(shape=(1,), dtype='int32', name='auxilary_input')
    aspect_embed = embedding_layer2(embeddings=embeddings, trainable=True)(auxilary_input)
    # bring the aspect embeddings in the appropriate format (we need to repeat the same aspect embedding k times)
    aspect_embed = keras.layers.Flatten(name='auxilary_flattened')(aspect_embed)
    aspect_embed = keras.layers.RepeatVector(max_length, name='auxilary_repeated')(aspect_embed)
    # concatenate the aspect embedding with the sent embedding
    concat = keras.layers.concatenate([embed, aspect_embed], name='concatenated_embeddings')
    concat = keras.layers.GaussianNoise(0.2)(concat)
    concat = keras.layers.Dropout(0.2, noise_shape=(1, 350), name='dropout_concat_emb', )(concat)
    concat = keras.layers.Dense(1024, activation='tanh', name='dense_after_emb')(concat)
    lstm = LSTM(rnn_size, return_sequences=True)(concat)
    hidden = keras.layers.concatenate([lstm, aspect_embed], name='concatenated_lstm_embedding')
    attention = Attention(name='attention')(hidden)
    atten_dropout = keras.layers.Dropout(0.2, noise_shape=(1, 178))(attention)
    output = Dense(classes, activation='softmax', name='dense_softmax')(atten_dropout)
    model = Functional_Model(inputs=[sent_input, auxilary_input], outputs=output)
    model.compile(optimizer=rmsprop(), loss=categorical_crossentropy, metrics=['accuracy'])
    print(model.summary())
    return model


def conc_emb_projection_dropout_LSTM_conc_emb_attention_projection(embeddings, classes, max_length):
    ''' BASELINE 7
     A simple model having a concatenation of word and aspect embeddings with gaussian
     noise,a projection layer with a tanh activation , a dropout layer,a LSTM cell,
     a concatenation of each hidden state with an aspect embedding, attention on top,
     a final dropout layer,a projection layer with a relu activation and a softmax for
      the final output'''

    rnn_size = 128
    # create sent embeddings
    sent_input = Input(shape=(max_length,), dtype='int32', name='sentence')
    embed = embedding_layer(embeddings=embeddings, max_length=max_length
                            , trainable=False)(sent_input)
    # create aspect embeddings
    auxilary_input = Input(shape=(1,), dtype='int32', name='auxilary_input')
    aspect_embed = embedding_layer2(embeddings=embeddings, trainable=True)(auxilary_input)
    # bring the aspect embeddings in the appropriate format (we need to repeat the same aspect embedding k times)
    aspect_embed = keras.layers.Flatten(name='auxilary_flattened')(aspect_embed)
    aspect_embed = keras.layers.RepeatVector(max_length, name='auxilary_repeated')(aspect_embed)
    # concatenate the aspect embedding with the sent embedding
    concat = keras.layers.concatenate([embed, aspect_embed], name='concatenated_embeddings')
    concat = keras.layers.GaussianNoise(0.2)(concat)
    concat = keras.layers.Dropout(0.2, noise_shape=(1, 350), name='dropout_concat_emb', )(concat)
    concat = keras.layers.Dense(1024, activation='tanh', name='dense_after_emb')(concat)
    lstm = LSTM(rnn_size, return_sequences=True)(concat)
    hidden = keras.layers.concatenate([lstm, aspect_embed], name='concatenated_lstm_embedding')
    attention = Attention(name='attention')(hidden)
    atten_dropout = keras.layers.Dropout(0.2, noise_shape=(1, 178))(attention)
    output = Dense(256, activation='relu', name='dense_relu')(atten_dropout)
    output = Dense(classes, activation='softmax', name='dense_softmax')(output)
    model = Functional_Model(inputs=[sent_input, auxilary_input], outputs=output)
    model.compile(optimizer=rmsprop(), loss=categorical_crossentropy, metrics=['accuracy'])
    print(model.summary())
    return model


def conc_emb_projection_dropout_Bi_LSTM_conc_emb_attention_projection(embeddings, classes, max_length):
    ''' BASELINE 8
     A simple model having a concatenation of word and aspect embeddings with gaussian
     noise,a projection layer with a tanh activation , a dropout layer,a Bi-LSTM cell,
     a concatenation of each hidden state with an aspect embedding, attention on top,
     a final dropout layer,a projection layer with a relu activation and a softmax for
      the final output'''

    rnn_size = 128
    # create sent embeddings
    sent_input = Input(shape=(max_length,), dtype='int32', name='sentence')
    embed = embedding_layer(embeddings=embeddings, max_length=max_length
                            , trainable=False)(sent_input)
    # create aspect embeddings
    auxilary_input = Input(shape=(1,), dtype='int32', name='auxilary_input')
    aspect_embed = embedding_layer2(embeddings=embeddings, trainable=True)(auxilary_input)
    # bring the aspect embeddings in the appropriate format (we need to repeat the same aspect embedding k times)
    aspect_embed = keras.layers.Flatten(name='auxilary_flattened')(aspect_embed)
    aspect_embed = keras.layers.RepeatVector(max_length, name='auxilary_repeated')(aspect_embed)
    # concatenate the aspect embedding with the sent embedding
    concat = keras.layers.concatenate([embed, aspect_embed], name='concatenated_embeddings')
    concat = keras.layers.GaussianNoise(0.2)(concat)
    concat = keras.layers.Dropout(0.2, noise_shape=(1, 350), name='dropout_concat_emb', )(concat)
    concat = keras.layers.Dense(1024, activation='tanh', name='dense_after_emb')(concat)
    lstm = Bidirectional(LSTM(rnn_size, return_sequences=True))(concat)
    hidden = keras.layers.concatenate([lstm, aspect_embed], name='concatenated_lstm_embedding')
    attention = Attention(name='attention')(hidden)
    atten_dropout = keras.layers.Dropout(0.2, noise_shape=(1, 306))(attention)
    output = Dense(256, activation='relu', name='dense_relu')(atten_dropout)
    output = Dense(classes, activation='softmax', name='dense_softmax')(output)
    model = Functional_Model(inputs=[sent_input, auxilary_input], outputs=output)
    model.compile(optimizer=rmsprop(), loss=categorical_crossentropy, metrics=['accuracy'])
    print(model.summary())
    return model


