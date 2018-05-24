def createCategories1():
    ''' We create all the possible categories 30 in total
    and we save each entity#attribute pair in a list'''

    entity_labels = ['RESTAURANT#', 'FOOD#', 'DRINKS#', 'AMBIENCE#', 'SERVICE#', 'LOCATION#']
    attribute_labels = ['GENERAL', 'PRICES', 'QUALITY', 'STYLE_OPTIONS', 'MISCELLANEOUS']

    entity_attribute_pairs = {}
    count = 0
    for entity in entity_labels:
        for attribute in attribute_labels:
            pair = entity + attribute
            entity_attribute_pairs[pair] = count
            count += 1
    return entity_attribute_pairs


def createCategories2():
    ''' We create only the categories that are encountered at least one time
    13 in total
    and we save each entity#attribute pair in a  list'''

    entity_attribute_pairs = {}
    # RESTAURANT
    entity_attribute_pairs['RESTAURANT#GENERAL'] = 0
    entity_attribute_pairs['RESTAURANT#PRICES'] = 1
    entity_attribute_pairs['RESTAURANT#MISCELLANEOUS'] = 2
    # FOOD
    entity_attribute_pairs['FOOD#PRICES'] = 3
    entity_attribute_pairs['FOOD#QUALITY'] = 4
    entity_attribute_pairs['FOOD#STYLE_OPTIONS'] = 5
    # DRINKS
    entity_attribute_pairs['DRINKS#PRICES'] = 6
    entity_attribute_pairs['DRINKS#QUALITY'] = 7
    entity_attribute_pairs['DRINKS#STYLE_OPTIONS'] = 8
    # AMBIENCE
    entity_attribute_pairs['AMBIENCE#GENERAL'] = 9
    # SERVICE
    entity_attribute_pairs['SERVICE#GENERAL'] = 10
    # LOCATION
    entity_attribute_pairs['LOCATION#GENERAL'] = 11
    # OTHER
    entity_attribute_pairs['OTHER'] = 12

    return entity_attribute_pairs

def createCategories3():
    ''' We create only the categories that are encountered at least one time
    13 in total
    and we save each entity#attribute pair in a  list'''

    entity_attribute_pairs = {}
    # RESTAURANT
    entity_attribute_pairs['RESTAURANT#GENERAL'] = 0

    # FOOD
    entity_attribute_pairs['FOOD#QUALITY'] = 1
    entity_attribute_pairs['FOOD#STYLE_OPTIONS'] = 2

    # SERVICE
    entity_attribute_pairs['SERVICE#GENERAL'] = 3

    entity_attribute_pairs['AMBIENCE#GENERAL'] = 4

    # OTHER
    entity_attribute_pairs['OTHER'] = 5

    return entity_attribute_pairs

def create_entitties_atributes():
    entity_labels = {}
    entity_labels['RESTAURANT'] = 0
    entity_labels['FOOD'] = 1
    entity_labels['DRINKS'] = 2
    entity_labels['AMBIENCE'] = 3
    entity_labels['SERVICE'] = 4
    entity_labels['LOCATION'] = 5

    attribute_labels = {}
    attribute_labels['GENERAL'] = 0
    attribute_labels['PRICES'] = 1
    attribute_labels['QUALITY'] = 2
    attribute_labels['STYLE_OPTIONS'] = 3
    attribute_labels['MISCELLANEOUS'] = 4

    return entity_labels,attribute_labels

def create_polarities():
    polarities = {}
    polarities['positive'] = 0
    polarities['negative'] = 1
    polarities['neutral'] = 2
    return polarities