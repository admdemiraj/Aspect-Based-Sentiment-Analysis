import numpy
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize.casual import casual_tokenize

oov = 0
total_words = 0

def binary_to_decimal(y_test):
    dec = numpy.flatnonzero(y_test)
    return dec


def extract_setiment_scores(sentences):
    ''' Method that given an array with sentences output the sentiment intensity
     of each sentence in a 3 way scale (positive,negative,neutral) '''

    sia = SentimentIntensityAnalyzer()
    sentiment_intersity = []
    for sent in sentences:
        temp = []
        intensity = sia.polarity_scores(sent)
        temp.append(intensity['pos'])
        temp.append(intensity['neg'])
        temp.append(intensity['neu'])
        sentiment_intersity.append(temp)
    return sentiment_intersity


def tokenize(text, lowercase=True):
    if lowercase:
        text = text.lower()
    return text.split()


def vectorize(text, word2idx, max_length, unk_policy="random"):
    """
    Covert array of tokens, to array of ids, with a fixed length
    and zero padding at the end
    Args:
        text (): the wordlist
        word2idx (): dictionary of word to ids
        max_length ():
        unk_policy (): how to handle OOV words

    Returns: list of ids with zero padding at the end

    """
    words = numpy.zeros(max_length).astype(int)

    # trim tokens after max length
    text = text[:max_length]
    global oov
    global total_words
    for i, token in enumerate(text):
        total_words += 1
        if token in word2idx:
            words[i] = word2idx[token]
        else:
            oov +=1
            if unk_policy == "random":
                words[i] = word2idx["<unk>"]
            elif unk_policy == "zero":
                words[i] = 0
    return words


def text_to_sequences(MAX_NB_WORDS, MAX_SEQUENCE_LENGTH, x_data):
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences

    # consider only the top e.g. 5000 most frequently occuring words from the data
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(x_data)

    # the tokenizer is holding 5000 words, if one of these words is encountered in the train dataset an index (0-5000)
    # is given to that word (the words not included in the tokenizer are ignored)
    sequences = tokenizer.texts_to_sequences(x_data)
    # make all sequences have a fixed length of indexes of words
    # if the have more keep only the first, if they have less add zeros to the empty indexes
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    return data

def sum_weight_ent_attributes(entities_weights,attribute_weights,categories= 13):
    ''' We have an array with 6 entities and another one with 5 attributes
    and we want to sum each entity with each attribute in order to create a weight for
    all the categories (E#A pairs)that are present'''
    sum_w = []
    for e_weight in entities_weights:
        for a_weight in attribute_weights:
            sum_w.append(sum(e_weight,a_weight))
    print(sum_w)


def prepare_text_data(text_array, word2idx, MAX_LENGTH ):
    "Method used for all the necessary text prepossessing before it enters the model"
    print("Tokenizing...")
    X_train = [casual_tokenize(x, preserve_case=False, reduce_len=True, strip_handles=False) for x in text_array]
    print("Vectorizing...")
    X_train = numpy.array([vectorize(x, word2idx, MAX_LENGTH) for x in X_train])
    print("Turning test and train data to numpy arrays")
    X_train = numpy.array(X_train)
    return X_train


def load_aspect_embeddings_from_file():
    file = '/home/admir/PycharmProjects/ABSA/embeddings/aspect_embeddings.txt'
    aspect_embeddings = []  # the word embeddings matrix

    # create the 2D array, which will be used for initializing
    # the Embedding layer of a NN.
    # We reserve the first row (idx=0), as the word embedding,
    # which will be used for zero padding (word with id = 0).
    aspect_embeddings.append(numpy.zeros(300))

    # read file, line by line
    with open(file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            values = line.split(" ")
            word = values[0]
            vector = numpy.asarray(values[1:], dtype='float32')
            aspect_embeddings.append(vector)

    aspect_embeddings = numpy.array(aspect_embeddings)
    return aspect_embeddings

load_aspect_embeddings_from_file()