import numpy
from keras_models.models import entities_attributes_projection_BiLSTM_attention_dencex2_dropout, entities_attributes_projection_LSTM_attention_dencex2_dropout
from load_data import read_xml2
from load_embeddings import load_word_vectors
from create_categories import createCategories2, create_entitties_atributes
from nlp import prepare_text_data
from Data_Augmentation_viaTranslation import create_classes_x_times, augment_data
from live_loss_plot import plots_slot1_t3
from sklearn.model_selection import train_test_split
from utilities import save_keras_model_JSON
from keras_models.metrics import evaluate_saved_models_slot1
import os

# get current directory
path = os.getcwd()
# get one directory up
path = os.path.dirname(path)

TRAIN_DATA = path+"/datasets/ABSA16_Restaurants_Train_SB1_v2.xml"
VAL_DATA = path+"/datasets/EN_REST_SB1_TEST.xml"
WORD_VECTORS = "../embeddings/word2vec.txt"

USE_TRANSLATION_AUGMENTATION = False
SPLIT_DATA_FOR_VALIDATION = False
VOTING_SYSTEMS = 1
EVALUATE_SAVED_MODELS = True

BATCH_SIZE = 128
EPOCHS = 50
WORD_VECTORS_DIMS = 300
MAX_LENGTH = 80


# load word embeddings
print("loading word embeddings...")
word2idx, idx2word, embeddings = load_word_vectors(WORD_VECTORS,
                                                   WORD_VECTORS_DIMS)
print("loading categories")
entity_attribute_pairs = createCategories2()
entity_labels, attribute_labels = create_entitties_atributes()
# load raw data
print("loading datasets...")

train_review, train_ent_attrib, train_entities, train_attributes = \
        read_xml2(entity_labels, attribute_labels, entity_attribute_pairs, TRAIN_DATA)

if SPLIT_DATA_FOR_VALIDATION:
    train_review, validation_review, train_ent_attrib, validation_ent_attrib,\
    train_entities, validation_entities, train_attributes, validation_attributes = \
        train_test_split(train_review, train_ent_attrib, train_entities, train_attributes, train_size=0.7)
    validation_entities = numpy.array(validation_entities)
    validation_ent_attrib = numpy.array(validation_ent_attrib)
    validation_attributes = numpy.array(validation_attributes)
    # preprocessing (Tokenizing, Vectorizing, Turning to Numpy arrays)
    validation_review = prepare_text_data(validation_review, word2idx, MAX_LENGTH)

if USE_TRANSLATION_AUGMENTATION:
    classes_x_times = create_classes_x_times()
    train_review, train_ent_attrib, train_entities, train_attributes = augment_data(train_review, train_ent_attrib, train_entities, train_attributes, classes_x_times)
    train_review = [str(i) for i in train_review]


gold_review, gold_ent_attrib, gold_entities, gold_attributes = \
    read_xml2(entity_labels, attribute_labels, entity_attribute_pairs, VAL_DATA)

y_train = train_ent_attrib
y_test = gold_ent_attrib

# preprocessing (Tokenizing, Vectorizing, Turning to Numpy arrays)
X_train = prepare_text_data(train_review, word2idx, MAX_LENGTH)
X_test = prepare_text_data(gold_review, word2idx, MAX_LENGTH)

print("Turning test and train data to numpy arrays")
y_train = numpy.array(y_train)
train_attributes = numpy.array(train_attributes)
train_entities = numpy.array(train_entities)
y_test = numpy.array(y_test)
gold_entities = numpy.array(gold_entities)
gold_attributes = numpy.array(gold_attributes)


if(EVALUATE_SAVED_MODELS):
    evaluate_saved_models_slot1(X_test, y_test)

classes = len(entity_attribute_pairs)


for i in range(0, VOTING_SYSTEMS):
    if SPLIT_DATA_FOR_VALIDATION:
        model = entities_attributes_projection_BiLSTM_attention_dencex2_dropout(embeddings=embeddings,
                                                                                max_length=MAX_LENGTH)

        history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                            validation_data=(validation_review, validation_ent_attrib))
        plots_slot1_t3(history=history)

    else:
        model = entities_attributes_projection_BiLSTM_attention_dencex2_dropout(embeddings=embeddings,
                                                                                max_length=MAX_LENGTH)

        history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)
        save_keras_model_JSON(model, str(i), path=path+"/saved_models/slot1/model")




