from googletrans import Translator
from utilities import load_cache_translation, write_cache_translation


def create_classes_x_times():
    classes_x_times = {}
    classes_x_times[0] = 1
    classes_x_times[1] = 1
    classes_x_times[2] = 1
    classes_x_times[3] = 1
    classes_x_times[4] = 1
    classes_x_times[5] = 1
    classes_x_times[6] = 1
    classes_x_times[7] = 1
    classes_x_times[8] = 1
    classes_x_times[9] = 1
    classes_x_times[10] = 1
    classes_x_times[11] = 1
    classes_x_times[12] = 1

    return classes_x_times


def augment_data(text, all_classes, entities, attributes, classes_x_times, strategy='single'):
    '''
    This is a method that does text augmentation by translating the
    given text in another language and then translating it back in
    the original one. The text will have changed a bit
    (it will be different) but hopefully similar with the original
     text.
    :param text:
    :param all_classes:
    :param classes_x_times:
    :param strategy:
    :return:
    text            --> the text we want to augment (and array of sentences)
    all_classes     --> the classes that are present in each sentence
    classes_x_times --> dictionary containing the classes we want to augment (usually minority classes)
                        and how many times (up to 3 because we get good
                        translations only between the languages: English,Spanish,German,French)
    strategy        --> single : translate from the original language to another and back to the original
                                 e.g. EN to DE to EN
                        double : translate from the original language to 2 other languages and back to the original
                                 e.g. EN to DE to SP to EN
    '''

    translator = Translator()
    # in order to avoid this time consuming operation, cache the results
    file = '/home/admir/PycharmProjects/ABSA/translation/translation.p'
    try:
        cache = load_cache_translation(file)
        print("Loaded translation from cache.")
        return cache
    except FileNotFoundError:
        pass

    return_sentences = []
    return_all_classes =[]
    return_entities = []
    return_attributes = []

    def translate_single(text, dest):
        result1 = translator.translate(text, dest=dest, src='en')
        result2 = translator.translate(result1.text, dest='en', src=dest)
        return str(result2.text)

    def translate_double(text, dest):
        result1 = translator.translate(text, dest=dest, src='en')
        result2 = translator.translate(result1.text, dest='de', src=dest)
        result3 = translator.translate(result2.text, dest='en', src='de')
        return str(result3.text)

    print(len(text))
    for idx, sent in enumerate(text):
        print(idx, sent)
        sent = str(sent)
        return_sentences.append(text)
        return_all_classes.append(all_classes[idx])
        return_entities.append(entities[idx])
        return_attributes.append(attributes[idx])
        category = (all_classes[idx] == 1).argmax(axis=0)
        if classes_x_times[category] == 1:
            translation = translate_single(sent,'de')
            return_sentences.append(translation)
            return_all_classes.append(all_classes[idx])
            return_entities.append(entities[idx])
            return_attributes.append(attributes[idx])
        elif classes_x_times[category] == 2:
            translation = translate_single(sent, 'de')
            return_sentences.append(translation)
            return_all_classes.append(all_classes[idx])
            return_entities.append(entities[idx])
            return_attributes.append(attributes[idx])

            translation = translate_single(sent, 'fr')
            return_sentences.append(translation)
            return_all_classes.append(all_classes[idx])
            return_entities.append(entities[idx])
            return_attributes.append(attributes[idx])
        elif classes_x_times[category] == 3:
            translation = translate_single(sent, 'de')
            return_sentences.append(translation)
            return_all_classes.append(all_classes[idx])
            return_entities.append(entities[idx])
            return_attributes.append(attributes[idx])

            translation = translate_single(sent, 'fr')
            return_sentences.append(translation)
            return_all_classes.append(all_classes[idx])
            return_entities.append(entities[idx])
            return_attributes.append(attributes[idx])

            translation = translate_single(sent, 'es')
            return_sentences.append(translation)
            return_all_classes.append(all_classes[idx])
            return_entities.append(entities[idx])
            return_attributes.append(attributes[idx])

    # write the data to a cache file
    write_cache_translation(file, (return_sentences, return_all_classes, return_entities, return_attributes))

    return return_sentences, return_all_classes, return_entities, return_attributes

def augment_data_polarity(text, polarity, aux, classes_x_times, strategy='single'):
    '''
    This is a method that does text augmentation by translating the
    given text in another language and then translating it back in
    the original one. The text will have changed a bit
    (it will be different) but hopefully similar with the original
     text.
    :param text:
    :param all_classes:
    :param classes_x_times:
    :param strategy:
    :return:
    text            --> the text we want to augment (and array of sentences)
    all_classes     --> the classes that are present in each sentence
    classes_x_times --> dictionary containing the classes we want to augment (usually minority classes)
                        and how many times (up to 3 because we get good
                        translations only between the languages: English,Spanish,German,French)
    strategy        --> single : translate from the original language to another and back to the original
                                 e.g. EN to DE to EN
                        double : translate from the original language to 2 other languages and back to the original
                                 e.g. EN to DE to SP to EN
    '''

    translator = Translator()
    # in order to avoid this time consuming operation, cache the results
    file = '/home/admir/PycharmProjects/ABSA/translation/translation_polarity.p'
    try:
        cache = load_cache_translation(file)
        print("Loaded translation from cache.")
        return cache
    except FileNotFoundError:
        pass

    return_sentences = []
    return_polarity = []
    return_aux = []

    def translate_single(text, dest):
        result1 = translator.translate(text, dest=dest, src='en')
        result2 = translator.translate(result1.text, dest='en', src=dest)
        return str(result2.text)

    def translate_double(text, dest):
        result1 = translator.translate(text, dest=dest, src='en')
        result2 = translator.translate(result1.text, dest='de', src=dest)
        result3 = translator.translate(result2.text, dest='en', src='de')
        return str(result3.text)

    print(len(text))
    for idx, sent in enumerate(text):
        print(idx, sent)
        sent = str(sent)
        return_sentences.append(text)
        return_polarity.append(polarity[idx])
        return_aux.append(aux[idx])
        category = (polarity[idx] == 1).argmax(axis=0)
        if classes_x_times[category] == 1:
            translation = translate_single(sent, 'de')
            return_sentences.append(translation)
            return_polarity.append(polarity[idx])
            return_aux.append(aux[idx])
        elif classes_x_times[category] == 2:
            translation = translate_single(sent, 'de')
            return_sentences.append(translation)
            return_polarity.append(polarity[idx])
            return_aux.append(aux[idx])


            translation = translate_single(sent, 'fr')
            return_sentences.append(translation)
            return_polarity.append(polarity[idx])
            return_aux.append(aux[idx])

        elif classes_x_times[category] == 3:
            translation = translate_single(sent, 'de')
            return_sentences.append(translation)
            return_polarity.append(polarity[idx])
            return_aux.append(aux[idx])


            translation = translate_single(sent, 'fr')
            return_sentences.append(translation)
            return_polarity.append(polarity[idx])
            return_aux.append(aux[idx])


            translation = translate_single(sent, 'es')
            return_sentences.append(translation)
            return_polarity.append(polarity[idx])
            return_aux.append(aux[idx])


    # write the data to a cache file
    write_cache_translation(file, (return_sentences, return_polarity, return_aux))

    return return_sentences, return_polarity, return_aux




