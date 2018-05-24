import numpy
from nlp import prepare_text_data
from keras_models.models import conc_emb_projection_dropout_Bi_LSTM_conc_emb_attention_projection, conc_emb_projection_dropout_LSTM_conc_emb_attention_projection
from keras_models.metrics import evaluate_saved_models_slot3
from live_loss_plot import plots
from load_data import read_xml_polarities
from load_embeddings import load_word_vectors
from create_categories import createCategories2, create_polarities
from sklearn.model_selection import train_test_split
from Data_Augmentation_viaTranslation import augment_data_polarity
from utilities import save_keras_model_JSON
import os

# get current directory
path = os.getcwd()
# get one directory up
path = os.path.dirname(path)

WORD_VECTORS = "../embeddings/word2vec.txt"
TRAIN_DATA = path+"/datasets/ABSA16_Restaurants_Train_SB1_v2.xml"
VAL_DATA = path+"/datasets/EN_REST_SB1_TEST.xml"

SPLIT_DATA_FOR_VALIDATION = False
USE_TRANSLATION_AUGMENTATION = False
EVALUATE_SAVED_MODELS = True
VOTING_SYSTEMS = 1

BATCH_SIZE = 128
EPOCHS = 50
MAX_LENGTH = 80
WORD_VECTORS_DIMS = 300



########################################################
# PREPARE FOR DATA
########################################################

# load word embeddings
print("loading word embeddings...")
word2idx, idx2word, embeddings = load_word_vectors(WORD_VECTORS,
                                                   WORD_VECTORS_DIMS)
print("loading categories")
entity_attribute_pairs = createCategories2()
polarity_labels = create_polarities()
# load raw data
print("loading datasets...")

train_review, train_ent_attrib, train_polarity, train_aux = \
    read_xml_polarities(entity_attribute_pairs, polarity_labels, TRAIN_DATA)

if SPLIT_DATA_FOR_VALIDATION:
    train_review, validation_review, train_ent_attrib, validation_ent_attrib, \
    train_polarity, validation_polarity, train_aux, validation_aux = \
    train_test_split(train_review, train_ent_attrib, train_polarity, train_aux, train_size=0.7)
    validation_ent_attrib = numpy.array(validation_ent_attrib)
    validation_polarity = numpy.array(validation_polarity)
    validation_aux = numpy.array(validation_aux)
    # preprocessing (Tokenizing, Vectorizing, Turning to Numpy arrays)
    validation_review = prepare_text_data(validation_review, word2idx, MAX_LENGTH)

gold_review, gold_ent_attrib, gold_polarity, gold_aux = \
    read_xml_polarities(entity_attribute_pairs, polarity_labels, VAL_DATA)


if USE_TRANSLATION_AUGMENTATION:
    classes_x_times = {}
    classes_x_times[0] = 1
    classes_x_times[1] = 1
    classes_x_times[2] = 1
    train_review, train_polarity, train_aux = augment_data_polarity(train_review, train_polarity, train_aux, classes_x_times)
    train_review = [str(i) for i in train_review]

y_train = train_polarity
y_test = gold_polarity

X_train = prepare_text_data(train_review, word2idx, MAX_LENGTH)
X_test = prepare_text_data(gold_review, word2idx, MAX_LENGTH)
words = gold_review

print("Turning test and train data to numpy arrays")
# train and test E#A labels
y_train = numpy.array(y_train)
y_test = numpy.array(y_test)
# auxilary input - the aspect that is present in the sentence
train_aux = numpy.array(train_aux)
gold_aux = numpy.array(gold_aux)


# entity and attribute pairs are given in this task and we can use them as extra features in each sentence
train_ent_attrib = numpy.array(train_ent_attrib)
gold_ent_attrib = numpy.array(gold_ent_attrib)

# evaluate on the models that are already trained and saved on disk
if EVALUATE_SAVED_MODELS:
    evaluate_saved_models_slot3(X_test, gold_aux, y_test, gold_review)


# MODEL
classes = len(polarity_labels)

model = conc_emb_projection_dropout_Bi_LSTM_conc_emb_attention_projection(embeddings, classes, MAX_LENGTH)

# train the model and save it on disk
for i in range(0, VOTING_SYSTEMS):
    if SPLIT_DATA_FOR_VALIDATION:
        history = model.fit([X_train, train_aux], y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=([validation_review, validation_aux], validation_polarity))
        plots(history=history)
    else:
        history = model.fit([X_train, train_aux], y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)
        save_keras_model_JSON(model, (str(i) + "_augmented"), path=path+"/saved_models/slot3/model")





