import xml.etree.ElementTree as ET
import numpy as np


def read_xml(entity_labels,attribute_labels,entity_attribute_pairs, xml_to_parse):
    # Method that reads the xml file with the data and returns an array of tuples, one tuple containing
    # the text and the other containing the labels in a binary format
    # e.g. [['Judging from previous posts this used to be a good place, but not any longer.'],
    # [0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,0.,  0.,  0.,  0.]]#
    # It has 0 when a label is not present and 1 when it is present





    aspects = np.zeros(shape=(len(entity_attribute_pairs)))
    number_of_aspects = len(entity_attribute_pairs)  # get the number of aspects
    tree = ET.parse(xml_to_parse)
    root = tree.getroot()
    data = []
    # read the reviews
    for review in root:
        data1 = []

        # read all sentences in the review
        for sentenses in review:

            # read each sentence
            for sentense in sentenses:
                new1 = []  # array containing two arrays (one for the sentences and one for the reviews)
                new2 = []  # array containing two arrays (one for the entities and one for the atributes)
                sent = sentense.find('text').text
                help_asp = np.zeros(shape=(number_of_aspects))  # reset the aspects to all being 0
                entities = np.zeros(shape=(len(entity_labels)))  # reset the entities to all being 0
                attributes = np.zeros(shape=(len(attribute_labels))) # reset the attributes to all being 0
                opinions = sentense.find('Opinions')

                # read each opinion in the sentence
                if opinions is not None:
                    for opinion in opinions:
                        tag = opinion.tag
                        attrib = opinion.attrib
                        category = attrib.get('category')
                        #split category into entity and attribute
                        e,a = category.split('#')
                        # check if entity is present
                        if e in entity_labels.keys():
                            entities[entity_labels[e]] = 1
                        # check if attribute is present
                        if a in attribute_labels.keys():
                            attributes[attribute_labels[a]] = 1
                        # if category is present get the index , else set it to 'OTHER' category
                        if category in entity_attribute_pairs.keys():
                            aspect_found_at = entity_attribute_pairs[category]
                        else:
                            aspect_found_at = 13  # 13 can be category 'OTHER'

                        if aspect_found_at is not None:
                            help_asp[aspect_found_at] = 1  # if an aspect is present turn 0 to 1
                            aspects[aspect_found_at] = aspects[
                                                           aspect_found_at] + 1  # find how many times each aspect appears
                new1.append(sent)
                new1.append(help_asp)
                new2.append(entities)
                new2.append(attributes)
                data1.append(new1)
                data1.append(new2)


        data.append(data1)

    return data





def read_xml2(entity_labels, attribute_labels, entity_attribute_pairs, xml_to_parse):
    # Method that reads the xml file with the data and returns an array of tuples, one tuple containing
    # the text and the other containing the labels in a binary format
    # e.g. [['Judging from previous posts this used to be a good place, but not any longer.'],
    # [0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,0.,  0.,  0.,  0.]]#
    # It has 0 when a label is not present and 1 when it is present

    aspects = np.zeros(shape=(len(entity_attribute_pairs)))
    number_of_aspects = len(entity_attribute_pairs)  # get the number of aspects
    tree = ET.parse(xml_to_parse)
    root = tree.getroot()
    data = []
    new1 = []  # array containing sentence
    new2 = []  # array containing entity#attribute pairs present in sentence in binary
    new3 = []  # array containing entities  present in sentence in binary
    new4 = []  # array containing attributes  present in sentence in binary
    # read the reviews
    for review in root:
        data1 = []

        # read all sentences in the review
        for sentenses in review:

            # read each sentence
            for sentense in sentenses:

                sent = sentense.find('text').text
                help_asp = np.zeros(shape=(number_of_aspects))  # reset the aspects to all being 0
                entities = np.zeros(shape=(len(entity_labels)))  # reset the entities to all being 0
                attributes = np.zeros(shape=(len(attribute_labels)))  # reset the attributes to all being 0
                opinions = sentense.find('Opinions')

                # read each opinion in the sentence
                if opinions is not None:
                    for opinion in opinions:
                        tag = opinion.tag
                        attrib = opinion.attrib
                        category = attrib.get('category')
                        # split category into entity and attribute
                        e, a = category.split('#')
                        # check if entity is present
                        if e in entity_labels.keys():
                            entities[entity_labels[e]] = 1
                        # check if attribute is present
                        if a in attribute_labels.keys():
                            attributes[attribute_labels[a]] = 1
                        # if category is present get the index , else set it to 'OTHER' category
                        if category in entity_attribute_pairs.keys():
                            aspect_found_at = entity_attribute_pairs[category]
                        else:
                            aspect_found_at = 13  # 13 can be category 'OTHER'

                        if aspect_found_at is not None:
                            help_asp[aspect_found_at] = 1  # if an aspect is present turn 0 to 1
                            aspects[aspect_found_at] = aspects[
                                                           aspect_found_at] + 1  # find how many times each aspect appears
                new1.append(sent)
                new2.append(help_asp)
                new3.append(entities)
                new4.append(attributes)

    return new1, new2, new3, new4
def read_xml2_train3(entity_attribute_pairs, xml_to_parse):
    tree = ET.parse(xml_to_parse)
    root = tree.getroot()
    new1 = []  # array containing sentence
    new3 = []  # array containing entities  present in sentence in binary
    # read the reviews
    for review in root:
        data1 = []

        # read all sentences in the review
        for sentenses in review:
            # read each sentence
            for sentense in sentenses:
                sent = sentense.find('text').text
                opinions = sentense.find('Opinions')
                # read each opinion in the sentence
                if opinions is not None:
                    for opinion in opinions:
                        attrib = opinion.attrib
                        category = attrib.get('category')
                        if category in entity_attribute_pairs.keys():
                            aspect_found_at = entity_attribute_pairs[category]
                        else:
                            aspect_found_at = 13  # 13 can be category 'OTHER'

                        if aspect_found_at is not None:

                            new3.append(aspect_found_at)
                            new1.append(sent)
    return new1,new3

def read_xml_polarities(entity_attribute_pairs,polarity_labels, xml_to_parse):
    # Method that reads the xml file with the data and returns an array of tuples, one tuple containing
    # the text and the other containing the labels in a binary format
    # e.g. [['Judging from previous posts this used to be a good place, but not any longer.'],
    # [0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,0.,  0.,  0.,  0.]]#
    # It has 0 when a label is not present and 1 when it is present


    number_of_aspects = len(entity_attribute_pairs)  # get the number of aspects
    tree = ET.parse(xml_to_parse)
    root = tree.getroot()
    new1 = []  # array containing sentence
    new2 = []  # array containing entity#attribute pairs present in sentence in binary
    new3 = []  # array containing polarity of entity attribute
    new4 = []  # array containing the index of the E#A pair
    # read the reviews
    for review in root:

        # read all sentences in the review
        for sentenses in review:

            # read each sentence
            for sentense in sentenses:
                sent = sentense.find('text').text
                opinions = sentense.find('Opinions')

                # read each opinion in the sentence
                if opinions is not None:
                    for opinion in opinions:
                        help_asp = np.zeros(shape=(number_of_aspects))  # reset the aspects to all being 0
                        polarity = np.zeros(shape=(len(polarity_labels)))  # reset the polarities to all being 0
                        attrib = opinion.attrib
                        category = attrib.get('category')
                        pol = attrib.get('polarity')
                        
                        # if category is present get the index , else set it to 'OTHER' category
                        if category in entity_attribute_pairs.keys():
                            aspect_found_at = entity_attribute_pairs[category]
                            polarity_position = polarity_labels[pol]

                        else:
                            aspect_found_at = 13  # 13 can be category 'OTHER'

                        if aspect_found_at is not None:
                            help_asp[aspect_found_at] = 1  # if an aspect is present turn 0 to 1
                            polarity[polarity_position] = 1 # if a polarity is present turn 0 to 1
                        new1.append(sent)
                        new2.append(help_asp)
                        new3.append(polarity)
                        new4.append(aspect_found_at)#return new4 and use it for aspect embedding
    return new1, new2, new3, new4


def get_words_for_each_category(entity_attribute_pairs,xml_to_parse):
    number_of_aspects = len(entity_attribute_pairs)  # get the number of aspects
    tree = ET.parse(xml_to_parse)
    root = tree.getroot()
    di = {}
    for i in range(0,12):
        di[i] = ' '

    # read the reviews
    for review in root:

        # read all sentences in the review
        for sentenses in review:

            # read each sentence
            for sentense in sentenses:
                sent = sentense.find('text').text
                opinions = sentense.find('Opinions')

                # read each opinion in the sentence
                if opinions is not None:
                    for opinion in opinions:

                        help_asp = np.zeros(shape=(number_of_aspects))  # reset the aspects to all being 0
                        attrib = opinion.attrib
                        category = attrib.get('category')
                        pol = attrib.get('polarity')
                        from_ = attrib.get('from')
                        from_ = int(from_)
                        to_ = attrib.get('to')
                        to_ = int(to_)
                        useful_text = sent[from_:to_]
                        category_number = entity_attribute_pairs[category]
                        di[category_number] = di[category_number]+' '+useful_text

    return di


